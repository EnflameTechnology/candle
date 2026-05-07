use crate::cpu_backend::CpuDevice;
use crate::{DType, Device, Result, WithDType};

#[cfg(any(feature = "cuda", feature = "gcu"))]
use half::{bf16, f16};

#[cfg(feature = "cuda")]
use crate::cuda::cudarc::driver::result;
#[cfg(feature = "cuda")]
use crate::cuda_backend::cudarc::driver::{DevicePtr, DeviceRepr};
#[cfg(feature = "cuda")]
use crate::cuda_backend::WrapErr;
#[cfg(feature = "cuda")]
use crate::cuda_backend::{CudaStorage, CudaStorageSlice};
#[cfg(feature = "cuda")]
use crate::{CpuStorageRef, Storage};
/// Host buffer for CPU offload / reload between device and pinned host memory.
#[derive(Clone, Debug)]
pub struct OffloadBuffer {
    ptr_host: *mut core::ffi::c_void,
    len: usize,
    pub cpu_device: CpuDevice,
    pub from_device: crate::Device,
    dtype: DType,
}

unsafe impl Send for OffloadBuffer {}
unsafe impl Sync for OffloadBuffer {}

#[cfg(feature = "cuda")]
pub fn storage_from_buffer<T: WithDType + DeviceRepr>(
    s: *mut T,
    len: usize,
    dev: &crate::CudaDevice,
) -> Result<CudaStorage> {
    let src = unsafe { std::slice::from_raw_parts(s, len) };
    let slice = match T::cpu_storage_ref(src) {
        CpuStorageRef::U8(_) => {
            let data = unsafe { dev.alloc::<u8>(len).w()? };
            unsafe {
                let _ = result::memcpy_htod_async(*data.device_ptr(), src, dev.cu_stream().clone());
            }
            CudaStorageSlice::U8(data)
        }
        CpuStorageRef::U32(_) => {
            let data = unsafe { dev.alloc::<u32>(len).w()? };
            unsafe {
                let _ = result::memcpy_htod_async(*data.device_ptr(), src, dev.cu_stream().clone());
            }
            CudaStorageSlice::U32(data)
        }
        CpuStorageRef::I64(_) => {
            let data = unsafe { dev.alloc::<i64>(len).w()? };
            unsafe {
                let _ = result::memcpy_htod_async(*data.device_ptr(), src, dev.cu_stream().clone());
            }
            CudaStorageSlice::I64(data)
        }
        CpuStorageRef::BF16(_) => {
            let data = unsafe { dev.alloc::<bf16>(len).w()? };
            unsafe {
                let _ = result::memcpy_htod_async(*data.device_ptr(), src, dev.cu_stream().clone());
            }
            CudaStorageSlice::BF16(data)
        }
        CpuStorageRef::F16(_) => {
            let data = unsafe { dev.alloc::<f16>(len).w()? };
            unsafe {
                let _ = result::memcpy_htod_async(*data.device_ptr(), src, dev.cu_stream().clone());
            }
            CudaStorageSlice::F16(data)
        }
        CpuStorageRef::F32(_) => {
            let data = unsafe { dev.alloc::<f32>(len).w()? };
            unsafe {
                let _ = result::memcpy_htod_async(*data.device_ptr(), src, dev.cu_stream().clone());
            }
            CudaStorageSlice::F32(data)
        }
        CpuStorageRef::F64(_) => {
            let data = unsafe { dev.alloc::<f64>(len).w()? };
            unsafe {
                let _ = result::memcpy_htod_async(*data.device_ptr(), src, dev.cu_stream().clone());
            }
            CudaStorageSlice::F64(data)
        }
    };
    Ok(crate::cuda_backend::CudaStorage {
        slice,
        device: dev.clone(),
    })
}

impl OffloadBuffer {
    pub fn new<T: WithDType>(
        src: &[T],
        dtype: DType,
        cpu_device: &CpuDevice,
        from_device: &Device,
    ) -> Result<Self> {
        let size = std::mem::size_of::<T>() * src.len();
        let mut ptr_host = std::ptr::null_mut();
        match from_device {
            #[cfg(feature = "gcu")]
            Device::Gcu(_) => {
                use crate::gcu_backend::ubridge::gcu_device::driv;
                unsafe {
                    driv::topsHostMalloc(&mut ptr_host as *mut *mut core::ffi::c_void, size, 0);
                    std::ptr::copy(src.as_ptr() as *mut core::ffi::c_void, ptr_host, size);
                }
            }
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => unsafe {
                ptr_host = result::malloc_host(size, 1).unwrap();
                std::ptr::copy(src.as_ptr() as *mut core::ffi::c_void, ptr_host, size);
            },
            _ => {
                crate::bail!("offload buffer only for cuda or gcu device tensors")
            }
        }
        Ok(OffloadBuffer {
            ptr_host,
            dtype,
            len: src.len(),
            cpu_device: cpu_device.clone(),
            from_device: from_device.clone(),
        })
    }

    pub fn reload(&self) -> Result<crate::Storage> {
        match &self.from_device {
            #[cfg(feature = "gcu")]
            Device::Gcu(dev) => {
                let storage = match self.dtype {
                    DType::BF16 => crate::Storage::Gcu(
                        dev.storage_from_buffer(self.ptr_host as *mut bf16, self.len)?,
                    ),
                    DType::F16 => crate::Storage::Gcu(
                        dev.storage_from_buffer(self.ptr_host as *mut f16, self.len)?,
                    ),
                    DType::F32 => crate::Storage::Gcu(
                        dev.storage_from_buffer(self.ptr_host as *mut f32, self.len)?,
                    ),
                    DType::U8 => crate::Storage::Gcu(
                        dev.storage_from_buffer(self.ptr_host as *mut u8, self.len)?,
                    ),
                    DType::U32 => crate::Storage::Gcu(
                        dev.storage_from_buffer(self.ptr_host as *mut u32, self.len)?,
                    ),
                    DType::I8 => crate::Storage::Gcu(
                        dev.storage_from_buffer(self.ptr_host as *mut i8, self.len)?,
                    ),
                    DType::I32 => crate::Storage::Gcu(
                        dev.storage_from_buffer(self.ptr_host as *mut i32, self.len)?,
                    ),
                    DType::I64 => crate::Storage::Gcu(
                        dev.storage_from_buffer(self.ptr_host as *mut i64, self.len)?,
                    ),
                    DType::F64 => crate::Storage::Gcu(
                        dev.storage_from_buffer(self.ptr_host as *mut f64, self.len)?,
                    ),
                };
                Ok(storage)
            }
            #[cfg(feature = "cuda")]
            Device::Cuda(dev) => {
                let storage = match self.dtype {
                    DType::BF16 => Storage::Cuda(storage_from_buffer(
                        self.ptr_host as *mut bf16,
                        self.len,
                        dev,
                    )?),
                    DType::F16 => Storage::Cuda(storage_from_buffer(
                        self.ptr_host as *mut f16,
                        self.len,
                        dev,
                    )?),
                    DType::F32 => Storage::Cuda(storage_from_buffer(
                        self.ptr_host as *mut f32,
                        self.len,
                        dev,
                    )?),
                    DType::U8 => Storage::Cuda(storage_from_buffer(
                        self.ptr_host as *mut u8,
                        self.len,
                        dev,
                    )?),
                    DType::U32 => Storage::Cuda(storage_from_buffer(
                        self.ptr_host as *mut u32,
                        self.len,
                        dev,
                    )?),
                    DType::I64 => Storage::Cuda(storage_from_buffer(
                        self.ptr_host as *mut i64,
                        self.len,
                        dev,
                    )?),
                    DType::F64 => Storage::Cuda(storage_from_buffer(
                        self.ptr_host as *mut f64,
                        self.len,
                        dev,
                    )?),
                };
                Ok(storage)
            }
            _ => crate::bail!("not supported device for cpu offloading"),
        }
    }
}

impl Drop for OffloadBuffer {
    fn drop(&mut self) {
        if self.ptr_host.is_null() {
            return;
        }
        match &self.from_device {
            #[cfg(feature = "gcu")]
            Device::Gcu(_) => {
                use crate::gcu_backend::ubridge::gcu_device::driv;
                unsafe {
                    driv::topsHostFree(self.ptr_host);
                }
            }
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => unsafe {
                let _ = result::free_host(self.ptr_host);
            },
            _ => {}
        }
    }
}
