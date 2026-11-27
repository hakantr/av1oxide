//! Resmî drop-in `libdav1d` (`1.5.4` / API `7.0.0`) C-ABI sistem öneki (`prefix/`) kurucusu.
//!
//! Kullanım:
//!   cargo run --locked --release -p av1oxide --example install_capi_prefix -- <prefix> [--target-dir <dir>]
//!
//! Hedef `<prefix>` dizinine aşağıdakileri kurar:
//! - `include/dav1d/{common,data,dav1d,headers,picture,version}.h` (6 resmî C başlığı)
//! - Linux: `lib/libdav1d.so.7.0.0`, `lib/libdav1d.so.7` -> `libdav1d.so.7.0.0`,
//!   `lib/libdav1d.so` -> `libdav1d.so.7` (`DT_SONAME = libdav1d.so.7`) ve `lib/libdav1d.a`
//! - macOS: `lib/libdav1d.7.dylib`, `lib/libdav1d.dylib` -> `libdav1d.7.dylib`
//!   (`install_name = @rpath/libdav1d.7.dylib`) ve `lib/libdav1d.a`
//! - `lib/pkgconfig/dav1d.pc` (`Version: 1.5.4`)

#![forbid(unsafe_code)]

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

const DAV1D_PC_VERSION: &str = "1.5.4";
const DAV1D_API_VERSION: &str = "7.0.0";
const DAV1D_SOVER: &str = "7";
const HEADER_NAMES: [&str; 6] = [
    "common.h",
    "data.h",
    "dav1d.h",
    "headers.h",
    "picture.h",
    "version.h",
];

fn print_usage() {
    eprintln!(
        "Kullanım: cargo run --locked --release -p av1oxide --example install_capi_prefix -- <prefix> [--target-dir <dir>]"
    );
}

fn absolutize(path: PathBuf) -> std::io::Result<PathBuf> {
    if path.is_absolute() {
        Ok(path)
    } else {
        Ok(env::current_dir()?.join(path))
    }
}

fn remove_existing_file_or_symlink(path: &Path) -> Result<(), String> {
    match fs::symlink_metadata(path) {
        Ok(meta) => {
            if meta.file_type().is_dir() {
                return Err(format!(
                    "hedef yolda beklenmeyen dizin çakışması var (silinmedi): {}",
                    path.display()
                ));
            }
            fs::remove_file(path)
                .map_err(|e| format!("mevcut dosya/bağ güncellenemedi ({}): {e}", path.display()))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(format!("hedef yol incelenemedi ({}): {e}", path.display())),
    }
}

fn install_file(src: &Path, dst: &Path) -> Result<(), String> {
    remove_existing_file_or_symlink(dst)?;
    fs::copy(src, dst)
        .map(|_| ())
        .map_err(|e| format!("{} -> {} kopyalanamadı: {e}", src.display(), dst.display()))
}

fn write_installed_file(dst: &Path, content: &str) -> Result<(), String> {
    remove_existing_file_or_symlink(dst)?;
    fs::write(dst, content).map_err(|e| format!("{} yazılamadı: {e}", dst.display()))
}

#[cfg(unix)]
fn create_symlink(target_rel: &str, link_path: &Path) -> Result<(), String> {
    remove_existing_file_or_symlink(link_path)?;
    std::os::unix::fs::symlink(target_rel, link_path)
        .map_err(|e| format!("{} bağı oluşturulamadı: {e}", link_path.display()))
}

#[cfg(not(unix))]
fn create_symlink(target_rel: &str, link_path: &Path) -> Result<(), String> {
    let src = link_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(target_rel);
    install_file(&src, link_path)
}

fn run() -> Result<(), String> {
    let mut prefix_arg: Option<PathBuf> = None;
    let mut target_dir_arg: Option<PathBuf> = None;
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_usage();
                return Ok(());
            }
            "--target-dir" => {
                let val = args
                    .next()
                    .ok_or_else(|| "--target-dir bayrağı bir dizin yolu gerektirir".to_string())?;
                target_dir_arg = Some(PathBuf::from(val));
            }
            other if other.starts_with('-') => {
                return Err(format!("bilinmeyen seçenek: {other}"));
            }
            positional => {
                if prefix_arg.is_some() {
                    return Err(format!("beklenmeyen ek konum argümanı: {positional}"));
                }
                prefix_arg = Some(PathBuf::from(positional));
            }
        }
    }

    let prefix = absolutize(
        prefix_arg
            .ok_or_else(|| "hedef <prefix> dizini zorunludur (yardım için --help)".to_string())?,
    )
    .map_err(|e| format!("önek yolu çözümlenemedi: {e}"))?;
    if prefix.parent().is_none() || prefix == Path::new("/") {
        return Err(format!(
            "güvenlik nedeniyle kök dizin (<prefix> = {}) hedef önek olarak kullanılamaz",
            prefix.display()
        ));
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .unwrap_or(manifest_dir.as_path())
        .to_path_buf();
    let header_src_dir = manifest_dir.join("include").join("dav1d");

    let target_dir = match target_dir_arg {
        Some(td) => absolutize(td).map_err(|e| format!("target-dir yolu çözümlenemedi: {e}"))?,
        None => workspace_root.join("target").join("capi_prefix"),
    };

    #[cfg(target_os = "macos")]
    let (soname, link_arg, built_cdylib, libs_private) = (
        format!("libdav1d.{DAV1D_SOVER}.dylib"),
        format!("-C link-arg=-Wl,-install_name,@rpath/libdav1d.{DAV1D_SOVER}.dylib"),
        "libav1oxide.dylib",
        "-lpthread -ldl -lm",
    );

    #[cfg(all(unix, not(target_os = "macos")))]
    let (soname, link_arg, built_cdylib, libs_private) = (
        format!("libdav1d.so.{DAV1D_SOVER}"),
        format!("-C link-arg=-Wl,-soname,libdav1d.so.{DAV1D_SOVER}"),
        "libav1oxide.so",
        "-lpthread -ldl -lm",
    );

    #[cfg(not(unix))]
    let (soname, link_arg, built_cdylib, libs_private) =
        ("dav1d.dll".to_string(), String::new(), "av1oxide.dll", "");

    let existing_rustflags = env::var("RUSTFLAGS").unwrap_or_default();
    let combined_rustflags = match (existing_rustflags.trim().is_empty(), link_arg.is_empty()) {
        (true, true) => String::new(),
        (true, false) => link_arg,
        (false, true) => existing_rustflags,
        (false, false) => format!("{} {link_arg}", existing_rustflags.trim()),
    };

    let cargo_bin = env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let mut cmd = Command::new(&cargo_bin);
    cmd.current_dir(&workspace_root)
        .arg("rustc")
        .arg("--locked")
        .arg("-p")
        .arg("av1oxide")
        .arg("--release")
        .arg("--features")
        .arg("capi")
        .arg("--target-dir")
        .arg(&target_dir)
        .arg("--crate-type")
        .arg("cdylib")
        .arg("--crate-type")
        .arg("staticlib");
    if !combined_rustflags.is_empty() {
        cmd.env("RUSTFLAGS", &combined_rustflags);
    }

    let status = cmd
        .status()
        .map_err(|e| format!("cargo rustc başlatılamadı: {e}"))?;
    if !status.success() {
        return Err(format!("cargo rustc başarısız oldu: {status}"));
    }

    let inc_dst = prefix.join("include").join("dav1d");
    let lib_dst = prefix.join("lib");
    let pc_dst = lib_dst.join("pkgconfig");
    fs::create_dir_all(&inc_dst).map_err(|e| format!("include dizini oluşturulamadı: {e}"))?;
    fs::create_dir_all(&pc_dst).map_err(|e| format!("pkgconfig dizini oluşturulamadı: {e}"))?;

    for hdr in HEADER_NAMES {
        let src = header_src_dir.join(hdr);
        if !src.is_file() {
            return Err(format!("eksik C başlık dosyası: {}", src.display()));
        }
        install_file(&src, &inc_dst.join(hdr))?;
    }

    let rel_dir = target_dir.join("release");
    let cdylib_src = rel_dir.join(built_cdylib);
    let static_src = rel_dir.join("libav1oxide.a");
    if !cdylib_src.is_file() {
        return Err(format!(
            "derlenen paylaşımlı kütüphane bulunamadı: {}",
            cdylib_src.display()
        ));
    }
    if !static_src.is_file() {
        return Err(format!(
            "derlenen statik kütüphane bulunamadı: {}",
            static_src.display()
        ));
    }

    #[cfg(target_os = "macos")]
    {
        install_file(&cdylib_src, &lib_dst.join(&soname))?;
        create_symlink(&soname, &lib_dst.join("libdav1d.dylib"))?;
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let full_so = format!("libdav1d.so.{DAV1D_API_VERSION}");
        install_file(&cdylib_src, &lib_dst.join(&full_so))?;
        create_symlink(&full_so, &lib_dst.join(&soname))?;
        create_symlink(&soname, &lib_dst.join("libdav1d.so"))?;
    }

    #[cfg(not(unix))]
    {
        install_file(&cdylib_src, &lib_dst.join(&soname))?;
    }

    install_file(&static_src, &lib_dst.join("libdav1d.a"))?;

    let prefix_str = prefix
        .to_str()
        .ok_or_else(|| "önek yolu geçerli UTF-8 değil".to_string())?;
    let pc_content = format!(
        "prefix={prefix_str}\n\
exec_prefix=${{prefix}}\n\
libdir=${{exec_prefix}}/lib\n\
includedir=${{prefix}}/include\n\
\n\
Name: libdav1d\n\
Description: AV1 decoding library (libdav1d {DAV1D_PC_VERSION} / API {DAV1D_API_VERSION} drop-in via av1oxide)\n\
Version: {DAV1D_PC_VERSION}\n\
Libs: -L${{libdir}} -ldav1d\n\
Libs.private: {libs_private}\n\
Cflags: -I${{includedir}}\n"
    );
    write_installed_file(&pc_dst.join("dav1d.pc"), &pc_content)?;

    println!(
        "av1oxide v{} drop-in libdav1d ({} / API {}) öneki kuruldu: {}",
        av1oxide::VERSION,
        DAV1D_PC_VERSION,
        DAV1D_API_VERSION,
        prefix.display()
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("hata: {msg}");
            ExitCode::FAILURE
        }
    }
}
