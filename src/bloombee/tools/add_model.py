import click
import json
from pathlib import Path
from bloombee.tools.model_analyzer import ModelAnalyzer
from bloombee.tools.code_generator import CodeGenerator

@click.command()
@click.option('--model-name', '-m', required=True, help='HuggingFace model name (e.g., mistralai/Mistral-7B-v0.1)')
@click.option('--output-dir', '-o', default='src/bloombee/models', help='Output directory')
@click.option('--dry-run', is_flag=True, help='只分析不生成')
@click.option('--show-metadata', is_flag=True, help='显示分析的元数据')
def add_model(model_name, output_dir, dry_run, show_metadata):
    """自动为 BloomBee 添加新模型支持"""
    
    click.echo(f"🔍 分析模型: {model_name}")
    
    # 分析模型
    analyzer = ModelAnalyzer()
    try:
        metadata = analyzer.analyze(model_name)
    except Exception as e:
        click.echo(f"❌ 分析失败: {e}", err=True)
        return
    
    click.echo(f"✅ 识别为 {metadata['model_name']} 模型")
    
    if show_metadata:
        click.echo("\n📊 模型元数据:")
        click.echo(json.dumps(metadata, indent=2))
    
    if dry_run:
        click.echo("\n🏃 Dry run 模式，跳过代码生成")
        return
    
    # 确认生成
    output_path = Path(output_dir) / metadata['model_name_lower']
    if output_path.exists():
        if not click.confirm(f"\n⚠️  目录 {output_path} 已存在，是否覆盖？"):
            return
    
    # 生成代码
    click.echo("\n🔨 生成代码...")
    generator = CodeGenerator()
    
    try:
        files = generator.generate(metadata, output_dir)
        
        click.echo("\n✅ 成功生成以下文件:")
        for f in files:
            click.echo(f"  - {f}")
        
        click.echo(f"\n🎉 完成！新模型已添加到 {output_path}")
        click.echo("\n下一步:")
        click.echo("1. 检查生成的代码，根据需要进行调整")
        click.echo("2. 测试模型加载和推理")
        click.echo("3. 如果有特殊功能，添加到相应文件中")
        
    except Exception as e:
        click.echo(f"\n❌ 生成失败: {e}", err=True)
        raise

if __name__ == '__main__':
    add_model()