from transformers import AutoConfig
from typing import Dict, Any
import re

class ModelAnalyzer:
    def analyze(self, model_name: str) -> Dict[str, Any]:
        """分析模型并提取模板所需的信息"""
        config = AutoConfig.from_pretrained(model_name)
        
        # 提取基本信息
        model_type = config.model_type
        architectures = config.architectures[0] if config.architectures else "Unknown"
        
        # 提取模型名称（如 Mistral, Llama, Qwen）
        model_class_name = self._extract_model_name(architectures)
        
        # 分析模型结构
        metadata = {
            # 基本信息
            "model_name": model_class_name,
            "model_name_lower": model_class_name.lower(),
            "transformers_model_type": model_type,
            "original_model_id": model_name,
            
            # 模型配置
            "num_hidden_layers": config.num_hidden_layers,
            "hidden_size": config.hidden_size,
            "num_attention_heads": config.num_attention_heads,
            
            # 检测特性
            "num_key_value_heads_attr": self._detect_kv_heads_attr(config),
            "block_prefix": self._detect_block_prefix(config),
            "needs_attention_mask": True,  # 大多数模型需要
            "needs_attention_mask_utils": True,
            "cache_reorder_pattern": self._detect_cache_pattern(config),
            "needs_pad_token": True,
            
            # 配置覆盖
            "config_overrides": self._get_config_overrides(config),
            
            # DHT设置
            "dht_prefix_suffix": "-hf" if "llama" in model_type else "",
            
            # License信息
            "license_info": self._get_license_info(model_type),
            
            # 模型属性映射
            "model_properties": self._get_model_properties(config),
            
            # Block属性
            "block_attributes": self._get_block_attributes(config),
            
            # 是否包含speculative模型
            "include_speculative": model_type in ["llama", "mistral"],
        }
        
        return metadata
    
    def _extract_model_name(self, architecture: str) -> str:
        """从架构名提取模型名"""
        # 例如: "MistralForCausalLM" -> "Mistral"
        match = re.match(r"(\w+)(?:Model|ForCausalLM|LMHeadModel)", architecture)
        return match.group(1) if match else "Unknown"
    
    def _detect_kv_heads_attr(self, config) -> str:
        """检测KV heads属性名"""
        if hasattr(config, "num_key_value_heads"):
            return "num_key_value_heads"
        elif hasattr(config, "n_kv_heads"):
            return "n_kv_heads"
        elif hasattr(config, "multi_query_attention"):
            return "multi_query_attention"
        return None
    
    def _detect_block_prefix(self, config) -> str:
        """检测块前缀"""
        # 常见模式
        if hasattr(config, "model_type"):
            if config.model_type in ["llama", "mistral", "qwen", "phi"]:
                return "model.layers"
            elif config.model_type in ["bloom", "falcon"]:
                return "transformer.h"
            elif config.model_type == "gpt2":
                return "transformer.h"
        return "model.layers"  # 默认值
    
    def _detect_cache_pattern(self, config) -> str:
        """检测缓存重排模式"""
        if config.model_type in ["llama", "mistral", "qwen"]:
            return "llama"
        elif config.model_type in ["bloom", "falcon"]:
            return "bloom"
        return "standard"
    
    def _get_config_overrides(self, config) -> Dict[str, Any]:
        """获取需要覆盖的配置"""
        overrides = {}
        if config.model_type in ["llama", "mistral"]:
            overrides["pretraining_tp"] = 1
            overrides["use_cache"] = True
        return overrides
    
    def _get_license_info(self, model_type: str) -> str:
        """获取许可信息"""
        licenses = {
            "llama": "Make sure you follow the Llama terms of use: https://llama.meta.com/llama3/license",
            "falcon": "Make sure you follow the Falcon terms of use",
            "bloom": "Make sure you follow the BLOOM terms of use: https://bit.ly/bloom-license",
        }
        return licenses.get(model_type, "")
    
    def _get_model_properties(self, config) -> Dict[str, str]:
        """获取模型属性映射"""
        properties = {}
        if config.model_type in ["llama", "mistral"]:
            properties.update({
                "word_embeddings": "self.embed_tokens",
                "word_embeddings_layernorm": "nn.Identity()",
                "h": "self.layers",
                "ln_f": "self.norm",
            })
        return properties
    
    def _get_block_attributes(self, config) -> list:
        """获取块需要的属性"""
        attrs = []
        if hasattr(config, "sliding_window"):
            attrs.append("sliding_window")
        if hasattr(config, "_attn_implementation"):
            attrs.append("_attn_implementation")
        return attrs