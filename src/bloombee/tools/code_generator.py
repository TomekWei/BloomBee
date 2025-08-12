import os
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
import black

class CodeGenerator:
    def __init__(self, template_dir: str = None):
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"
        
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )
    
    def generate(self, metadata: dict, output_dir: str) -> list:
        """生成所有模型文件"""
        model_name_lower = metadata["model_name_lower"]
        output_path = Path(output_dir) / model_name_lower
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        # 生成每个文件
        templates = [
            ("__init__.py.j2", "__init__.py"),
            ("config.py.j2", "config.py"),
            ("block.py.j2", "block.py"),
            ("model.py.j2", "model.py"),
        ]
        
        for template_name, output_name in templates:
            content = self._render_template(template_name, metadata)
            
            # 格式化代码
            try:
                content = black.format_str(content, mode=black.Mode())
            except Exception as e:
                print(f"Warning: Failed to format {output_name}: {e}")
            
            # 写入文件
            output_file = output_path / output_name
            with open(output_file, "w") as f:
                f.write(content)
            
            generated_files.append(str(output_file))
        
        return generated_files
    
    def _render_template(self, template_name: str, context: dict) -> str:
        """渲染单个模板"""
        template = self.env.get_template(template_name)
        return template.render(**context)