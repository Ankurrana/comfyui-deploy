"""
Tests for the WorkflowParser module.
"""
import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from comfyui_deploy.workflow_parser import (
    WorkflowParser,
    WorkflowDependencies,
    ModelReference,
    CustomNodeReference,
)


# Sample minimal workflow for testing
SAMPLE_WORKFLOW = {
    "nodes": [
        {
            "id": 1,
            "type": "CheckpointLoaderSimple",
            "properties": {"cnr_id": "comfy-core"},
            "widgets_values": ["sd_xl_base_1.0.safetensors"]
        },
        {
            "id": 2,
            "type": "LoraLoader",
            "properties": {"cnr_id": "comfy-core"},
            "widgets_values": ["my_lora.safetensors", 1.0, 1.0]
        },
        {
            "id": 3,
            "type": "WanVideoModelLoader",
            "properties": {
                "cnr_id": "ComfyUI-WanVideoWrapper",
                "ver": "abc123"
            },
            "widgets_values": ["LongCat_model.safetensors", "bf16"]
        },
        {
            "id": 4,
            "type": "Label (rgthree)",
            "properties": {},  # No cnr_id!
            "widgets_values": []
        },
        {
            "id": 5,
            "type": "VHS_VideoCombine",
            "properties": {"cnr_id": "comfyui-videohelpersuite"},
            "widgets_values": {}
        },
    ]
}


class TestWorkflowParser:
    """Test cases for WorkflowParser."""
    
    @pytest.fixture
    def parser(self):
        """Create a parser instance."""
        return WorkflowParser()
    
    @pytest.fixture
    def mock_node_db(self):
        """Mock the node database to avoid network calls."""
        with patch('comfyui_deploy.workflow_parser.get_node_database') as mock:
            db = MagicMock()
            db.get_repo_for_node_type.return_value = None
            db.extract_cnr_id_from_url.return_value = "rgthree-comfy"
            mock.return_value = db
            yield mock
    
    def test_parse_data_extracts_models(self, parser, mock_node_db):
        """Test that models are extracted correctly."""
        deps = parser.parse_data(SAMPLE_WORKFLOW)
        
        # Should find checkpoint, lora, and diffusion model
        assert len(deps.models) == 3
        
        filenames = [m.filename for m in deps.models]
        assert "sd_xl_base_1.0.safetensors" in filenames
        assert "my_lora.safetensors" in filenames
        assert "LongCat_model.safetensors" in filenames
    
    def test_parse_data_extracts_custom_nodes(self, parser, mock_node_db):
        """Test that custom nodes are extracted correctly."""
        deps = parser.parse_data(SAMPLE_WORKFLOW)
        
        # Should find WanVideoWrapper and VideoHelperSuite (not comfy-core)
        cnr_ids = [n.cnr_id for n in deps.custom_nodes]
        
        assert "ComfyUI-WanVideoWrapper" in cnr_ids
        assert "comfyui-videohelpersuite" in cnr_ids
        assert "comfy-core" not in cnr_ids
    
    def test_parse_data_detects_rgthree_by_pattern(self, parser, mock_node_db):
        """Test that rgthree nodes are detected by pattern fallback."""
        deps = parser.parse_data(SAMPLE_WORKFLOW)
        
        cnr_ids = [n.cnr_id for n in deps.custom_nodes]
        assert "rgthree-comfy" in cnr_ids
    
    def test_model_type_mapping(self, parser, mock_node_db):
        """Test that model types are mapped to correct folders."""
        deps = parser.parse_data(SAMPLE_WORKFLOW)
        
        model_map = {m.filename: m for m in deps.models}
        
        # Checkpoint -> checkpoints folder
        assert model_map["sd_xl_base_1.0.safetensors"].target_folder == "models/checkpoints"
        
        # LoRA -> loras folder
        assert model_map["my_lora.safetensors"].target_folder == "models/loras"
        
        # WanVideo model -> diffusion_models folder
        assert model_map["LongCat_model.safetensors"].target_folder == "models/diffusion_models"
    
    def test_invalid_model_names_filtered(self, parser, mock_node_db):
        """Test that invalid model names are filtered out."""
        workflow = {
            "nodes": [
                {
                    "id": 1,
                    "type": "CheckpointLoaderSimple",
                    "properties": {"cnr_id": "comfy-core"},
                    "widgets_values": ["none"]  # Invalid name
                },
                {
                    "id": 2,
                    "type": "LoraLoader",
                    "properties": {"cnr_id": "comfy-core"},
                    "widgets_values": [""]  # Empty name
                },
            ]
        }
        
        deps = parser.parse_data(workflow)
        assert len(deps.models) == 0
    
    def test_github_url_from_cnr_id(self, parser, mock_node_db):
        """Test that GitHub URLs are resolved from cnr_id."""
        deps = parser.parse_data(SAMPLE_WORKFLOW)
        
        wan_node = next(n for n in deps.custom_nodes if n.cnr_id == "ComfyUI-WanVideoWrapper")
        assert wan_node.github_url == "https://github.com/kijai/ComfyUI-WanVideoWrapper"
    
    def test_parse_empty_workflow(self, parser, mock_node_db):
        """Test parsing an empty workflow."""
        deps = parser.parse_data({"nodes": []})
        
        assert len(deps.models) == 0
        assert len(deps.custom_nodes) == 0
    
    def test_parse_file(self, parser, mock_node_db, tmp_path):
        """Test parsing from a file."""
        workflow_file = tmp_path / "test_workflow.json"
        workflow_file.write_text(json.dumps(SAMPLE_WORKFLOW))
        
        deps = parser.parse(workflow_file)
        
        assert len(deps.models) == 3
        assert len(deps.custom_nodes) >= 2


class TestModelReference:
    """Test cases for ModelReference dataclass."""
    
    def test_model_reference_equality(self):
        """Test that model references with same filename are equal."""
        ref1 = ModelReference(
            filename="model.safetensors",
            node_type="Loader",
            node_id=1,
            model_type="checkpoint",
            target_folder="models/checkpoints"
        )
        ref2 = ModelReference(
            filename="model.safetensors",
            node_type="OtherLoader",
            node_id=2,
            model_type="checkpoint",
            target_folder="models/checkpoints"
        )
        
        # Same filename = same model
        assert ref1 == ref2
        assert hash(ref1) == hash(ref2)


class TestCustomNodeReference:
    """Test cases for CustomNodeReference dataclass."""
    
    def test_custom_node_equality(self):
        """Test that custom node refs with same cnr_id are equal."""
        ref1 = CustomNodeReference(
            cnr_id="my-node",
            version="1.0",
            node_types=["NodeA"],
            github_url="https://github.com/user/repo"
        )
        ref2 = CustomNodeReference(
            cnr_id="my-node",
            version="2.0",  # Different version
            node_types=["NodeB"],  # Different types
            github_url=None  # No URL
        )
        
        # Same cnr_id = same package
        assert ref1 == ref2
        assert hash(ref1) == hash(ref2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
