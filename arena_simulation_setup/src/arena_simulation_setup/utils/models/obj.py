from pathlib import Path

import aiofiles

from . import Model, ModelProvider, ModelType


class ModelProvider_OBJ(ModelProvider.provides(ModelType.OBJ)):
    @classmethod
    async def load(cls, model_dir: Path, model: str, loader_args: dict | None) -> Model:
        model_paths = (
            model_dir / f"{model}.obj" / f"{model}.obj",
            model_dir / f"{model}.obj",
        )
        model_path = next((p for p in model_paths if p.is_file()), None)
        if model_path is None:
            raise FileNotFoundError(f"Could not find OBJ model file for '{model}' in '{model_dir}' (searched: {model_paths})")
        async with aiofiles.open(model_path) as f:
            return Model(type=ModelType.OBJ, name=model, description=await f.read(), path=model_path)
