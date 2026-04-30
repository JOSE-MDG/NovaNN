use crate::handle::RustHandle;
use crate::ops;
use std::ffi::c_void;

#[unsafe(no_mangle)]
pub extern "C" fn reserve(size: usize, align: usize) -> RustHandle {
    ops::reserve_op(size, align).unwrap_or_else(|_| RustHandle::invalid())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn retain(handle: *mut RustHandle) {
    if handle.is_null() {
        return;
    }
    let _ = ops::retain_op(unsafe { &*handle });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn release(handle: *mut RustHandle) {
    if handle.is_null() {
        return;
    }
    let _ = ops::release_op(unsafe { &*handle });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn resize(handle: *mut RustHandle, new_size: usize) -> bool {
    if handle.is_null() {
        return false;
    }
    ops::resize_op(unsafe { &mut *handle }, new_size).is_ok()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_data_from(handle: *mut RustHandle) -> *mut c_void {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    ops::get_data_op(unsafe { &*handle }).unwrap_or(std::ptr::null_mut()) as *mut c_void
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn is_valid_handle(handle: *mut RustHandle) -> bool {
    if handle.is_null() {
        return false;
    }
    ops::is_valid_op(unsafe { &*handle })
}
