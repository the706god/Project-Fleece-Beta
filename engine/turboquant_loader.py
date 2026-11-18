"""
Self-contained Native TurboQuant Runtime & Ctypes Struct ABI Loader.
Automatically binds the repository-bundled libllama / libmtmd binaries and
aligns llama_cpp C-struct definitions (llama_model_params / llama_context_params)
without requiring any host system or virtualenv modifications.
"""

import importlib.util
import os
import platform
import sys


def configure_turboquant_runtime() -> str:
    """Bind bundled libllama/libmtmd dylibs and inject TurboQuant ABI ctypes shim."""
    engine_dir = os.path.dirname(os.path.realpath(__file__))
    arch_tag = f"{sys.platform}-{platform.machine().lower()}"
    if arch_tag == "darwin-arm64":
        bin_dir = os.path.join(engine_dir, "bin", "darwin-arm64")
    else:
        bin_dir = os.path.join(engine_dir, "bin", arch_tag)

    libllama_path = os.path.join(bin_dir, "libllama.dylib")
    libmtmd_path = os.path.join(bin_dir, "libmtmd.dylib")

    if os.path.exists(libllama_path):
        os.environ.setdefault("LLAMA_CPP_LIB_PATH", bin_dir)
        os.environ.setdefault("LLAMA_CPP_LIB", libllama_path)
        if os.path.exists(libmtmd_path):
            os.environ.setdefault("MTMD_CPP_LIB_PATH", bin_dir)
            # Note: llama_cpp.mtmd_cpp expects MTMD_CPP_LIB to be the DIRECTORY containing libmtmd.
            existing_mtmd = os.environ.get("MTMD_CPP_LIB")
            if not existing_mtmd or existing_mtmd.endswith((".dylib", ".so", ".dll")):
                os.environ["MTMD_CPP_LIB"] = bin_dir

        # Inject bundled TurboQuant-aligned ctypes bindings before llama_cpp loads
        shim_path = os.path.join(engine_dir, "turboquant_llama_cpp.py")
        if os.path.exists(shim_path) and "llama_cpp.llama_cpp" not in sys.modules:
            try:
                import llama_cpp._ctypes_extensions  # noqa: F401 - ensure package exists

                spec = importlib.util.spec_from_file_location("llama_cpp.llama_cpp", shim_path)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules["llama_cpp.llama_cpp"] = mod
                    spec.loader.exec_module(mod)
            except Exception:
                pass
    return bin_dir
