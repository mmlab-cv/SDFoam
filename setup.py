import os
import platform
import re
import shutil
import sys
from pathlib import Path

import cmake_build_extension
import setuptools
import subprocess

source_dir = Path(__file__).parent.absolute()


def _venv_python(venv):
    scripts_dir = "Scripts" if os.name == "nt" else "bin"
    executable = "python.exe" if os.name == "nt" else "python"
    return Path(venv) / scripts_dir / executable


def _python_purelib(python):
    return Path(
        subprocess.check_output(
            [
                str(python),
                "-c",
                "import sysconfig; print(sysconfig.get_path('purelib'))",
            ],
            text=True,
        ).strip()
    )


def _target_python_and_purelib():
    candidates = []
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        candidates.append(_venv_python(venv))

    candidates.append(_venv_python(source_dir / ".venv"))
    candidates.append(Path(sys.executable))

    for python in candidates:
        if not python.exists():
            continue
        purelib = _python_purelib(python)
        if (purelib / "torch").exists():
            return python, purelib

    return Path(sys.executable), _python_purelib(Path(sys.executable))


python_executable, lib_path = _target_python_and_purelib()
python_prefix = python_executable.parent.parent
assert (
    lib_path / "torch"
).exists(), (
    "Could not find PyTorch in the target Python environment; please install "
    "PyTorch before installing sdfoam."
)

cmake_options = []

if "CUDA_HOME" in os.environ:
    cmake_options.append(f"-DCUDA_TOOLKIT_ROOT_DIR={os.environ['CUDA_HOME']}")

cmake_generator = (
    os.environ.get("SDFOAM_CMAKE_GENERATOR")
    or os.environ.get("CMAKE_GENERATOR")
    or ("Visual Studio 17 2022" if platform.system() == "Windows" else "Ninja")
)

cmake = (source_dir / "CMakeLists.txt").read_text()
version = re.search(r"project\(\S+ VERSION (\S+)\)", cmake).group(1)


class SDFoamBuildExtension(cmake_build_extension.BuildExtension):
    def build_extension(self, ext):
        build_folder = Path(".").absolute() / f"{self.build_temp}_{ext.name}"
        cache = build_folder / "CMakeCache.txt"
        if cache.exists():
            cached_generator = None
            for line in cache.read_text(errors="ignore").splitlines():
                if line.startswith("CMAKE_GENERATOR:INTERNAL="):
                    cached_generator = line.split("=", 1)[1]
                    break

            if cached_generator and cached_generator != ext.cmake_generator:
                shutil.rmtree(build_folder)

        super().build_extension(ext)

install_requirements = [
    "cmake==3.29.2",
    "cmake-format",
    "cmake_build_extension",
    "ConfigArgParse",
    "einops",
    "glfw==2.6.5",
    "pycolmap",
    "opencv-python",
    "pillow",
    "plyfile",
    "pybind11[global]",
    "pyyaml",
    "scipy",
    "scikit-learn",
    "tqdm",
    "wandb",
    "open3d",
]


setuptools.setup(
    version=version,
    install_requires=install_requirements,
    ext_modules=[
        cmake_build_extension.CMakeExtension(
            name="SDFoamBindings",
            install_prefix="sdfoam",
            cmake_depends_on=["pybind11"],
            write_top_level_init=None,
            source_dir=str(source_dir),
            cmake_generator=cmake_generator,
            cmake_configure_options=[
                f"-DPython3_EXECUTABLE={python_executable}",
                f"-DPython3_ROOT_DIR={python_prefix}",
                "-DCALL_FROM_SETUP_PY:BOOL=ON",
                "-DBUILD_SHARED_LIBS:BOOL=OFF",
                "-DGPU_DEBUG:BOOL=OFF",
                "-DEXAMPLE_WITH_PYBIND11:BOOL=ON",
                f"-DTorch_DIR={lib_path}/torch/share/cmake/Torch",
                "-DPIP_GLFW:BOOL=ON",
            ]
            + cmake_options,
        ),
    ],
    cmdclass=dict(
        build_ext=SDFoamBuildExtension,
    ),
)
