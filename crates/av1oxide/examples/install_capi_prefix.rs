//! Resmî drop-in `libdav1d` (`1.5.4` / API `7.0.0`) C-ABI sistem öneki (`prefix/`) kurucusu.
//!
//! Kullanım:
//!   cargo run --locked --release -p av1oxide --example install_capi_prefix -- <prefix> [--target-dir <dir>]
//!
//! Hedef `<prefix>` dizinine aşağıdakileri işlemsel (ön-doğrulama + sahneleme + atomik `rename` +
//! hata anında tam geri alma) olarak kurar:
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

enum InstallAction {
    CopyFile {
        src: PathBuf,
        dst: PathBuf,
    },
    WriteText {
        content: String,
        dst: PathBuf,
    },
    #[cfg_attr(not(unix), allow(dead_code))]
    Symlink {
        target_rel: String,
        dst: PathBuf,
    },
}

impl InstallAction {
    fn dst(&self) -> &Path {
        match self {
            Self::CopyFile { dst, .. }
            | Self::WriteText { dst, .. }
            | Self::Symlink { dst, .. } => dst,
        }
    }
}

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

fn ensure_dir_candidate_ok(dir: &Path) -> Result<(), String> {
    match fs::symlink_metadata(dir) {
        Ok(meta) => {
            if meta.file_type().is_dir() {
                Ok(())
            } else if meta.file_type().is_symlink() {
                match fs::metadata(dir) {
                    Ok(target_meta) if target_meta.is_dir() => Ok(()),
                    Ok(_) => Err(format!(
                        "hedef dizin yolu dizin olmayan bir sembolik bağdır (önek değiştirilmedi): {}",
                        dir.display()
                    )),
                    Err(e) => Err(format!(
                        "hedef dizin sembolik bağı çözümlenemedi (önek değiştirilmedi) ({}): {e}",
                        dir.display()
                    )),
                }
            } else {
                Err(format!(
                    "hedef dizin yolu mevcut bir dosyayla çakışıyor (önek değiştirilmedi): {}",
                    dir.display()
                ))
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(format!(
            "hedef dizin yolu incelenemedi (önek değiştirilmedi) ({}): {e}",
            dir.display()
        )),
    }
}

fn ensure_file_target_not_dir(path: &Path) -> Result<(), String> {
    match fs::symlink_metadata(path) {
        Ok(meta) => {
            if meta.file_type().is_dir() {
                Err(format!(
                    "hedef yolda beklenmeyen dizin çakışması var (önek değiştirilmedi): {}",
                    path.display()
                ))
            } else {
                Ok(())
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(format!(
            "hedef yol incelenemedi (önek değiştirilmedi) ({}): {e}",
            path.display()
        )),
    }
}

fn sibling_temp_path(dst: &Path, tag: &str, pid: u32, idx: usize) -> Result<PathBuf, String> {
    let parent = dst.parent().ok_or_else(|| {
        format!(
            "hedef yolun ebeveyn dizini belirlenemedi: {}",
            dst.display()
        )
    })?;
    let name = dst
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| format!("hedef dosya adı geçerli UTF-8 değil: {}", dst.display()))?;
    Ok(parent.join(format!(".av1oxide_{tag}_{pid}_{idx}_{name}")))
}

fn stage_action(action: &InstallAction, stage_path: &Path) -> Result<(), String> {
    if fs::symlink_metadata(stage_path).is_ok() {
        let _ = fs::remove_file(stage_path);
    }
    let res = match action {
        InstallAction::CopyFile { src, .. } => fs::copy(src, stage_path).map(|_| ()).map_err(|e| {
            format!(
                "{} -> {} sahnelenemedi: {e}",
                src.display(),
                stage_path.display()
            )
        }),
        InstallAction::WriteText { content, .. } => fs::write(stage_path, content)
            .map_err(|e| format!("{} sahnelenemedi: {e}", stage_path.display())),
        InstallAction::Symlink { target_rel, dst } => {
            #[cfg(unix)]
            {
                let _ = dst;
                std::os::unix::fs::symlink(target_rel, stage_path).map_err(|e| {
                    format!("{} sembolik bağı sahnelenemedi: {e}", stage_path.display())
                })
            }
            #[cfg(not(unix))]
            {
                let src = dst
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .join(target_rel);
                fs::copy(&src, stage_path).map(|_| ()).map_err(|e| {
                    format!(
                        "{} -> {} kopyalanamadı: {e}",
                        src.display(),
                        stage_path.display()
                    )
                })
            }
        }
    };
    if res.is_err() {
        let _ = fs::remove_file(stage_path);
    }
    res
}

fn rollback_transaction(
    activated: &[PathBuf],
    backups: &[(PathBuf, PathBuf)],
    staged: &[PathBuf],
    created_dirs: &[PathBuf],
) {
    for dst in activated.iter().rev() {
        let _ = fs::remove_file(dst);
    }
    for (dst, bak) in backups.iter().rev() {
        let _ = fs::rename(bak, dst);
    }
    for stage_path in staged {
        let _ = fs::remove_file(stage_path);
    }
    for dir in created_dirs.iter().rev() {
        let _ = fs::remove_dir(dir);
    }
}

fn execute_transactional_install(
    required_dirs: &[PathBuf],
    actions: &[InstallAction],
) -> Result<(), String> {
    // 1. Ön-doğrulama: tek bir bayt yazmadan veya dizin oluşturmadan tüm yolları denetle.
    for dir in required_dirs {
        ensure_dir_candidate_ok(dir)?;
    }
    for action in actions {
        ensure_file_target_not_dir(action.dst())?;
    }

    // 2. Eksik dizinleri (gerekiyorsa `<prefix>` üst ata dizinleri dahil) oluştur ve geri alma için kaydet.
    let mut all_dirs: Vec<PathBuf> = Vec::new();
    if let Some(first_dir) = required_dirs.first() {
        let mut missing_ancestors: Vec<PathBuf> = Vec::new();
        let mut cur = first_dir.parent();
        while let Some(p) = cur {
            if fs::symlink_metadata(p).is_ok() {
                ensure_dir_candidate_ok(p)?;
                break;
            }
            missing_ancestors.push(p.to_path_buf());
            cur = p.parent();
        }
        for anc in missing_ancestors.into_iter().rev() {
            all_dirs.push(anc);
        }
    }
    all_dirs.extend(required_dirs.iter().cloned());

    let mut created_dirs: Vec<PathBuf> = Vec::new();
    for dir in &all_dirs {
        if fs::symlink_metadata(dir).is_err() {
            if let Err(e) = fs::create_dir(dir) {
                for created in created_dirs.iter().rev() {
                    let _ = fs::remove_dir(created);
                }
                return Err(format!(
                    "hedef dizin oluşturulamadı ({}): {e}",
                    dir.display()
                ));
            }
            created_dirs.push(dir.clone());
        }
    }

    // 3. Sahneleme: tüm yeni dosyaları ve bağları aynı dizinde geçici kardeş adlarla hazırla.
    let pid = std::process::id();
    let mut staged_paths: Vec<PathBuf> = Vec::with_capacity(actions.len());
    for (idx, action) in actions.iter().enumerate() {
        let stage_path = match sibling_temp_path(action.dst(), "stage", pid, idx) {
            Ok(p) => p,
            Err(err) => {
                rollback_transaction(&[], &[], &staged_paths, &created_dirs);
                return Err(err);
            }
        };
        staged_paths.push(stage_path.clone());
        if let Err(err) = stage_action(action, &stage_path) {
            rollback_transaction(&[], &[], &staged_paths, &created_dirs);
            return Err(err);
        }
    }

    // 4. Atomik devreye alma: mevcut hedefleri yedekleyip sahnelenmiş dosyaları `rename` ile geçir.
    let mut backups: Vec<(PathBuf, PathBuf)> = Vec::new();
    let mut activated: Vec<PathBuf> = Vec::new();
    for (idx, (action, stage_path)) in actions.iter().zip(staged_paths.iter()).enumerate() {
        let dst = action.dst();
        match fs::symlink_metadata(dst) {
            Ok(meta) => {
                if meta.file_type().is_dir() {
                    rollback_transaction(&activated, &backups, &staged_paths, &created_dirs);
                    return Err(format!(
                        "hedef yolda beklenmeyen dizin çakışması var (geri alındı): {}",
                        dst.display()
                    ));
                }
                let bak_path = match sibling_temp_path(dst, "bak", pid, idx) {
                    Ok(p) => p,
                    Err(err) => {
                        rollback_transaction(&activated, &backups, &staged_paths, &created_dirs);
                        return Err(err);
                    }
                };
                if fs::symlink_metadata(&bak_path).is_ok() {
                    let _ = fs::remove_file(&bak_path);
                }
                if let Err(e) = fs::rename(dst, &bak_path) {
                    rollback_transaction(&activated, &backups, &staged_paths, &created_dirs);
                    return Err(format!(
                        "mevcut hedef yedeklenemedi (geri alındı) ({}): {e}",
                        dst.display()
                    ));
                }
                backups.push((dst.to_path_buf(), bak_path));
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                rollback_transaction(&activated, &backups, &staged_paths, &created_dirs);
                return Err(format!(
                    "hedef yol incelenemedi (geri alındı) ({}): {e}",
                    dst.display()
                ));
            }
        }
        if let Err(e) = fs::rename(stage_path, dst) {
            rollback_transaction(&activated, &backups, &staged_paths, &created_dirs);
            return Err(format!(
                "{} -> {} devreye alınamadı (geri alındı): {e}",
                stage_path.display(),
                dst.display()
            ));
        }
        activated.push(dst.to_path_buf());
    }

    // 5. İşlem tamamlandı: geçici yedekleri temizle.
    for (_, bak_path) in backups {
        let _ = fs::remove_file(bak_path);
    }
    Ok(())
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

    // Derlemeye başlamadan önce hedef yolları hızlıca ön-doğrula.
    let inc_root = prefix.join("include");
    let inc_dst = inc_root.join("dav1d");
    let lib_dst = prefix.join("lib");
    let pc_dst = lib_dst.join("pkgconfig");
    let required_dirs = vec![
        prefix.clone(),
        inc_root,
        inc_dst.clone(),
        lib_dst.clone(),
        pc_dst.clone(),
    ];

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

    let rel_dir = target_dir.join("release");
    let cdylib_src = rel_dir.join(built_cdylib);
    let static_src = rel_dir.join("libav1oxide.a");

    let mut actions: Vec<InstallAction> = Vec::new();
    for hdr in HEADER_NAMES {
        let src = header_src_dir.join(hdr);
        if !src.is_file() {
            return Err(format!("eksik C başlık dosyası: {}", src.display()));
        }
        actions.push(InstallAction::CopyFile {
            src,
            dst: inc_dst.join(hdr),
        });
    }

    #[cfg(target_os = "macos")]
    {
        actions.push(InstallAction::CopyFile {
            src: cdylib_src.clone(),
            dst: lib_dst.join(&soname),
        });
        actions.push(InstallAction::Symlink {
            target_rel: soname.clone(),
            dst: lib_dst.join("libdav1d.dylib"),
        });
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let full_so = format!("libdav1d.so.{DAV1D_API_VERSION}");
        actions.push(InstallAction::CopyFile {
            src: cdylib_src.clone(),
            dst: lib_dst.join(&full_so),
        });
        actions.push(InstallAction::Symlink {
            target_rel: full_so,
            dst: lib_dst.join(&soname),
        });
        actions.push(InstallAction::Symlink {
            target_rel: soname.clone(),
            dst: lib_dst.join("libdav1d.so"),
        });
    }

    #[cfg(not(unix))]
    {
        actions.push(InstallAction::CopyFile {
            src: cdylib_src.clone(),
            dst: lib_dst.join(&soname),
        });
    }

    actions.push(InstallAction::CopyFile {
        src: static_src.clone(),
        dst: lib_dst.join("libdav1d.a"),
    });
    actions.push(InstallAction::WriteText {
        content: pc_content,
        dst: pc_dst.join("dav1d.pc"),
    });

    // Derleme öncesinde de dizin/çakışma doğrulaması yap ki hatalı önekte boşuna derlenmesin.
    for dir in &required_dirs {
        ensure_dir_candidate_ok(dir)?;
    }
    for action in &actions {
        ensure_file_target_not_dir(action.dst())?;
    }

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

    execute_transactional_install(&required_dirs, &actions)?;

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
