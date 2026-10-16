//! Test support: fabricate a SNAC model directory without the real weights.

use anyhow::Result;
use candle_core::{DType, Device, Result as CResult, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::{Init, VarBuilder};
use candle_transformers::models::snac;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::engines::maya1_llamacpp::{SNAC_CONFIG_FILE, SNAC_WEIGHTS_FILE};

/// The SNAC 24 kHz architecture (hubertsiuzdak/snac_24khz `config.json`).
pub const SNAC_TEST_CONFIG: &str = r#"{"sampling_rate":24000,"encoder_dim":48,"encoder_rates":[2,4,8,8],
    "decoder_dim":768,"decoder_rates":[8,8,4,2],"attn_window_size":32,"codebook_size":4096,
    "codebook_dim":8,"vq_strides":[4,2,1],"noise":true,"depthwise":true}"#;

#[derive(Clone, Default)]
struct Recorder(Arc<Mutex<HashMap<String, Tensor>>>);

impl SimpleBackend for Recorder {
    fn get(&self, s: Shape, name: &str, _: Init, dt: DType, dev: &Device) -> CResult<Tensor> {
        // candle's weight-norm convs first ask for a dense (out, in, k) weight and fall back to the
        // depthwise (out, 1, k) shape when a real checkpoint rejects the first. A backend that invents
        // tensors would accept the dense shape and silently build a much heavier, wrong model, so
        // reject it where the model is depthwise: size-7 kernels with equal in/out channels.
        if let [out, inp, 7] = s.dims() {
            if name.ends_with("original1") && out == inp && *out > 1 {
                return Err(candle_core::Error::Msg("dense shape rejected: depthwise layer".into()).bt());
            }
        }
        let t = Tensor::zeros(s, dt, dev)?;
        self.0.lock().unwrap().insert(name.to_string(), t.clone());
        Ok(t)
    }
    fn get_unchecked(&self, name: &str, _: DType, _: &Device) -> CResult<Tensor> {
        Err(candle_core::Error::CannotFindTensor {
            path: name.to_string(),
        }
        .bt())
    }
    fn contains_tensor(&self, name: &str) -> bool {
        self.0.lock().unwrap().contains_key(name)
    }
}

/// Writes `config.json` and a correctly named and shaped, all-zero `snac_24khz.safetensors` into
/// `dir`, by building the model once with a backend that records every tensor it asks for. The audio
/// it decodes is meaningless; the shapes, file loading and names are real.
pub fn write_zero_weight_snac(dir: &Path) -> Result<()> {
    let cfg: snac::Config = serde_json::from_str(SNAC_TEST_CONFIG)?;
    let recorder = Recorder::default();
    let vb = VarBuilder::from_backend(Box::new(recorder.clone()), DType::F32, Device::Cpu);
    snac::Model::new(&cfg, vb)?;
    let tensors = recorder.0.lock().unwrap().clone();
    std::fs::write(dir.join(SNAC_CONFIG_FILE), SNAC_TEST_CONFIG)?;
    candle_core::safetensors::save(&tensors, dir.join(SNAC_WEIGHTS_FILE))?;
    Ok(())
}

/// The real SNAC 24 kHz architecture with all-zero weights (meaningless audio, true cost and shapes).
pub fn zero_weight_model() -> Result<snac::Model> {
    let cfg: snac::Config = serde_json::from_str(SNAC_TEST_CONFIG)?;
    let vb = VarBuilder::from_backend(Box::new(Recorder::default()), DType::F32, Device::Cpu);
    Ok(snac::Model::new(&cfg, vb)?)
}
