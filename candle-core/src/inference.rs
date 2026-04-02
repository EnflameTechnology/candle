use std::cell::Cell;

thread_local! {
    static INFERENCE_MODE_DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// Scoped guard that marks the current thread as running an inference-only
/// workload.
///
/// Backends can use this to enable more memory-efficient allocation behavior
/// during model loading and inference.
///
/// ```rust
/// let _guard = candle_core::InferenceMode::enter();
/// // inference-only / model loading code
/// ```
#[derive(Debug)]
pub struct InferenceMode {
    active: bool,
}

impl InferenceMode {
    pub fn enter() -> Self {
        INFERENCE_MODE_DEPTH.with(|depth| depth.set(depth.get() + 1));
        Self { active: true }
    }

    pub fn is_enabled() -> bool {
        INFERENCE_MODE_DEPTH.with(|depth| depth.get() > 0)
    }

    pub fn with<T>(f: impl FnOnce() -> T) -> T {
        let _guard = Self::enter();
        f()
    }
}

impl Drop for InferenceMode {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        INFERENCE_MODE_DEPTH.with(|depth| {
            let current = depth.get();
            debug_assert!(current > 0, "InferenceMode depth underflow");
            depth.set(current.saturating_sub(1));
        });
        self.active = false;
    }
}
