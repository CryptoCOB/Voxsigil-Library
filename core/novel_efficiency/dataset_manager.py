"""
Data Management and Licensing Infrastructure for Novel LLM Paradigms
Addresses data & licensing compliance and dataset versioning risks

Comprehensive data management with:
- Dataset licensing validation
- Version control for reproducibility 
- Compliance checking for academic datasets
- ARC-AGI specific data handling
- Logical reasoning ground truth management

Part of HOLO-1.5 Recursive Symbolic Cognition Mesh
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from pathlib import Path
import aiofiles
from datetime import datetime, timezone
from urllib.request import urlopen
import zipfile
import tempfile

try:
    from ...agents.base import vanta_agent, CognitiveMeshRole, BaseAgent
    HOLO_AVAILABLE = True
except ImportError:
    # Fallback for non-HOLO environments
    HOLO_AVAILABLE = False
    def vanta_agent(*args, **kwargs):
        def decorator(cls):
            return cls
        return decorator
    
    class CognitiveMeshRole:
        PROCESSOR = "processor"
    
    class BaseAgent:
        def __init__(self, *args, **kwargs):
            pass
        
        async def async_init(self):
            pass


class LicenseType(Enum):
    """Supported license types for datasets"""
    MIT = "MIT"
    APACHE_2_0 = "Apache-2.0"
    BSD_3_CLAUSE = "BSD-3-Clause"
    GPL_V3 = "GPL-v3"
    CC_BY_4_0 = "CC-BY-4.0"
    CC_BY_SA_4_0 = "CC-BY-SA-4.0"
    CC0_1_0 = "CC0-1.0"
    ACADEMIC_ONLY = "Academic-Only"
    PROPRIETARY = "Proprietary"
    UNKNOWN = "Unknown"


class DatasetType(Enum):
    """Types of datasets for paradigm training"""
    ARC_AGI = "arc-agi"
    LOGICAL_REASONING = "logical-reasoning"
    VISUAL_REASONING = "visual-reasoning"
    MATHEMATICAL = "mathematical"
    LINGUISTIC = "linguistic"
    MULTIMODAL = "multimodal"
    SYNTHETIC = "synthetic"
    EVALUATION = "evaluation"


class ComplianceLevel(Enum):
    """Compliance verification levels"""
    VERIFIED = "verified"        # Fully verified license and usage
    APPROVED = "approved"        # Pre-approved by legal team
    PENDING = "pending"          # Under review
    RESTRICTED = "restricted"    # Use restrictions apply
    BLOCKED = "blocked"          # Cannot be used


@dataclass
class DatasetLicense:
    """Dataset licensing information"""
    license_type: LicenseType
    license_url: Optional[str] = None
    attribution_required: bool = True
    commercial_use: bool = False
    derivative_works: bool = True
    distribution: bool = True
    modification: bool = True
    private_use: bool = True
    
    # Academic use specific
    academic_only: bool = False
    citation_required: bool = True
    paper_reference: Optional[str] = None
    
    # Additional constraints
    restrictions: List[str] = field(default_factory=list)
    expiry_date: Optional[datetime] = None


@dataclass
class DatasetMetadata:
    """Comprehensive dataset metadata"""
    name: str
    version: str
    dataset_type: DatasetType
    license: DatasetLicense
    
    # Source information
    source_url: Optional[str] = None
    download_url: Optional[str] = None
    paper_url: Optional[str] = None
    github_url: Optional[str] = None
    
    # Content description
    description: str = ""
    num_samples: Optional[int] = None
    size_gb: Optional[float] = None
    format: str = "json"
    
    # Quality metrics
    validation_accuracy: Optional[float] = None
    human_verified: bool = False
    automated_checks: List[str] = field(default_factory=list)
    
    # Version control
    checksum_sha256: Optional[str] = None
    created_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Compliance
    compliance_level: ComplianceLevel = ComplianceLevel.PENDING
    compliance_notes: str = ""
    reviewed_by: Optional[str] = None
    review_date: Optional[datetime] = None


@dataclass
class LogicalGroundTruth:
    """Ground truth data for logical reasoning tasks"""
    task_id: str
    task_type: str  # e.g., "propositional", "predicate", "temporal"
    
    # Input representation
    premises: List[str]
    query: str
    
    # Expected output
    conclusion: bool
    reasoning_steps: List[str]
    logical_form: Optional[str] = None
    
    # Difficulty metrics
    complexity_level: int = 1  # 1-5 scale
    reasoning_depth: int = 1
    num_entities: int = 0
    num_relations: int = 0
    
    # Validation
    verified: bool = False
    verification_method: str = ""
    confidence_score: float = 1.0


class DatasetRegistry:
    """Registry for tracking all datasets and their compliance status"""
    
    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self.datasets: Dict[str, DatasetMetadata] = {}
        self.compliance_cache: Dict[str, ComplianceLevel] = {}
        self.logger = logging.getLogger(__name__)
    
    async def load_registry(self):
        """Load dataset registry from file"""
        if self.registry_path.exists():
            async with aiofiles.open(self.registry_path, 'r') as f:
                content = await f.read()
                registry_data = json.loads(content)
                
                for dataset_id, data in registry_data.items():
                    # Reconstruct license object
                    license_data = data["license"]
                    license_obj = DatasetLicense(
                        license_type=LicenseType(license_data["license_type"]),
                        license_url=license_data.get("license_url"),
                        attribution_required=license_data.get("attribution_required", True),
                        commercial_use=license_data.get("commercial_use", False),
                        derivative_works=license_data.get("derivative_works", True),
                        distribution=license_data.get("distribution", True),
                        modification=license_data.get("modification", True),
                        private_use=license_data.get("private_use", True),
                        academic_only=license_data.get("academic_only", False),
                        citation_required=license_data.get("citation_required", True),
                        paper_reference=license_data.get("paper_reference"),
                        restrictions=license_data.get("restrictions", [])
                    )
                    
                    # Reconstruct metadata object
                    metadata = DatasetMetadata(
                        name=data["name"],
                        version=data["version"],
                        dataset_type=DatasetType(data["dataset_type"]),
                        license=license_obj,
                        source_url=data.get("source_url"),
                        download_url=data.get("download_url"),
                        paper_url=data.get("paper_url"),
                        github_url=data.get("github_url"),
                        description=data.get("description", ""),
                        num_samples=data.get("num_samples"),
                        size_gb=data.get("size_gb"),
                        format=data.get("format", "json"),
                        validation_accuracy=data.get("validation_accuracy"),
                        human_verified=data.get("human_verified", False),
                        automated_checks=data.get("automated_checks", []),
                        checksum_sha256=data.get("checksum_sha256"),
                        compliance_level=ComplianceLevel(data.get("compliance_level", "pending")),
                        compliance_notes=data.get("compliance_notes", ""),
                        reviewed_by=data.get("reviewed_by"),
                    )
                    
                    self.datasets[dataset_id] = metadata
                
                self.logger.info(f"Loaded {len(self.datasets)} datasets from registry")
    
    async def save_registry(self):
        """Save dataset registry to file"""
        registry_data = {}
        
        for dataset_id, metadata in self.datasets.items():
            # Convert to serializable format
            license_dict = {
                "license_type": metadata.license.license_type.value,
                "license_url": metadata.license.license_url,
                "attribution_required": metadata.license.attribution_required,
                "commercial_use": metadata.license.commercial_use,
                "derivative_works": metadata.license.derivative_works,
                "distribution": metadata.license.distribution,
                "modification": metadata.license.modification,
                "private_use": metadata.license.private_use,
                "academic_only": metadata.license.academic_only,
                "citation_required": metadata.license.citation_required,
                "paper_reference": metadata.license.paper_reference,
                "restrictions": metadata.license.restrictions
            }
            
            registry_data[dataset_id] = {
                "name": metadata.name,
                "version": metadata.version,
                "dataset_type": metadata.dataset_type.value,
                "license": license_dict,
                "source_url": metadata.source_url,
                "download_url": metadata.download_url,
                "paper_url": metadata.paper_url,
                "github_url": metadata.github_url,
                "description": metadata.description,
                "num_samples": metadata.num_samples,
                "size_gb": metadata.size_gb,
                "format": metadata.format,
                "validation_accuracy": metadata.validation_accuracy,
                "human_verified": metadata.human_verified,
                "automated_checks": metadata.automated_checks,
                "checksum_sha256": metadata.checksum_sha256,
                "compliance_level": metadata.compliance_level.value,
                "compliance_notes": metadata.compliance_notes,
                "reviewed_by": metadata.reviewed_by,
                "created_date": metadata.created_date.isoformat(),
                "updated_date": metadata.updated_date.isoformat()
            }
        
        async with aiofiles.open(self.registry_path, 'w') as f:
            await f.write(json.dumps(registry_data, indent=2))
        
        self.logger.info(f"Saved {len(self.datasets)} datasets to registry")
    
    async def register_dataset(self, dataset_id: str, metadata: DatasetMetadata):
        """Register a new dataset"""
        self.datasets[dataset_id] = metadata
        await self.save_registry()
        self.logger.info(f"Registered dataset: {dataset_id}")
    
    async def update_compliance(self, dataset_id: str, level: ComplianceLevel, 
                               notes: str = "", reviewer: str = ""):
        """Update dataset compliance status"""
        if dataset_id in self.datasets:
            self.datasets[dataset_id].compliance_level = level
            self.datasets[dataset_id].compliance_notes = notes
            self.datasets[dataset_id].reviewed_by = reviewer
            self.datasets[dataset_id].review_date = datetime.now(timezone.utc)
            await self.save_registry()
            self.logger.info(f"Updated compliance for {dataset_id}: {level.value}")
    
    def get_compliant_datasets(self, dataset_type: Optional[DatasetType] = None) -> List[str]:
        """Get list of compliant datasets"""
        compliant = []
        for dataset_id, metadata in self.datasets.items():
            if metadata.compliance_level in [ComplianceLevel.VERIFIED, ComplianceLevel.APPROVED]:
                if dataset_type is None or metadata.dataset_type == dataset_type:
                    compliant.append(dataset_id)
        return compliant
    
    def check_license_compatibility(self, dataset_id: str, intended_use: str) -> bool:
        """Check if dataset license is compatible with intended use"""
        if dataset_id not in self.datasets:
            return False
        
        license_info = self.datasets[dataset_id].license
        
        # Check basic usage rights
        if intended_use == "commercial" and not license_info.commercial_use:
            return False
        
        if intended_use == "academic" and license_info.academic_only:
            return True
        
        if intended_use == "derivative" and not license_info.derivative_works:
            return False
        
        if intended_use == "distribution" and not license_info.distribution:
            return False
        
        return True


@vanta_agent(role=CognitiveMeshRole.PROCESSOR)
class DatasetManager(BaseAgent):
    """
    Comprehensive Dataset Management System
    
    Handles dataset licensing, compliance, versioning, and ground truth management
    for Novel LLM Paradigms with focus on ARC-AGI and logical reasoning datasets.
    """
    
    def __init__(self, data_directory: Path, registry_path: Optional[Path] = None):
        super().__init__()
        self.data_directory = Path(data_directory)
        self.registry_path = registry_path or (self.data_directory / "dataset_registry.json")
        
        # Initialize components
        self.registry = DatasetRegistry(self.registry_path)
        self.logical_ground_truth: Dict[str, List[LogicalGroundTruth]] = {}
        
        # Predefined dataset configurations
        self.predefined_datasets = self._initialize_predefined_datasets()
        
        # Cognitive metrics for HOLO-1.5
        self.cognitive_metrics = {
            "compliance_rate": 0.0,
            "data_integrity": 1.0,
            "licensing_risk": 0.0,
            "ground_truth_coverage": 0.0
        }
        
        self.logger = logging.getLogger(__name__)
    
    async def async_init(self):
        """Initialize dataset manager and register with HOLO-1.5 mesh"""
        if HOLO_AVAILABLE:
            await super().async_init()
            await self.register_capabilities([
                "dataset_management",
                "license_compliance",
                "data_versioning",
                "ground_truth_validation"
            ])
        
        # Ensure data directory exists
        self.data_directory.mkdir(parents=True, exist_ok=True)
        
        # Load existing registry
        await self.registry.load_registry()
        
        # Register predefined datasets
        await self._register_predefined_datasets()
        
        # Load logical ground truth data
        await self._load_logical_ground_truth()
        
        self.logger.info(f"DatasetManager initialized with {len(self.registry.datasets)} datasets")
    
    def _initialize_predefined_datasets(self) -> Dict[str, DatasetMetadata]:
        """Initialize predefined dataset configurations"""
        datasets = {}
        
        # ARC-AGI Official Dataset
        datasets["arc-agi-official"] = DatasetMetadata(
            name="ARC-AGI Official Dataset",
            version="1.0",
            dataset_type=DatasetType.ARC_AGI,
            license=DatasetLicense(
                license_type=LicenseType.APACHE_2_0,
                license_url="https://github.com/fchollet/ARC-AGI/blob/main/LICENSE",
                commercial_use=True,
                attribution_required=True,
                citation_required=True,
                paper_reference="Chollet, F. (2019). The Measure of Intelligence. arXiv:1911.01547"
            ),
            source_url="https://github.com/fchollet/ARC-AGI",
            download_url="https://github.com/fchollet/ARC-AGI/archive/refs/heads/main.zip",
            paper_url="https://arxiv.org/abs/1911.01547",
            github_url="https://github.com/fchollet/ARC-AGI",
            description="Official Abstraction and Reasoning Corpus (ARC) dataset for AGI evaluation",
            num_samples=800,  # 400 training + 400 test
            size_gb=0.01,
            format="json",
            human_verified=True,
            compliance_level=ComplianceLevel.VERIFIED
        )
        
        # ConceptARC Dataset
        datasets["concept-arc"] = DatasetMetadata(
            name="ConceptARC",
            version="1.0", 
            dataset_type=DatasetType.ARC_AGI,
            license=DatasetLicense(
                license_type=LicenseType.MIT,
                commercial_use=True,
                attribution_required=True,
                citation_required=True,
                paper_reference="Moskvichev, A. et al. (2023). The ConceptARC Benchmark"
            ),
            description="Extended ARC-style tasks with explicit concept annotations",
            num_samples=300,
            size_gb=0.005,
            format="json",
            compliance_level=ComplianceLevel.APPROVED
        )
        
        # CLUTRR Dataset (for logical reasoning)
        datasets["clutrr"] = DatasetMetadata(
            name="CLUTRR",
            version="1.0",
            dataset_type=DatasetType.LOGICAL_REASONING,
            license=DatasetLicense(
                license_type=LicenseType.MIT,
                commercial_use=True,
                attribution_required=True,
                citation_required=True,
                paper_reference="Sinha, K. et al. (2019). CLUTRR: A Diagnostic Benchmark for Inductive Reasoning from Text"
            ),
            source_url="https://github.com/facebookresearch/clutrr",
            paper_url="https://arxiv.org/abs/1908.06177",
            description="Compositional Language Understanding with Text-based Relational Reasoning",
            num_samples=10000,
            size_gb=0.1,
            format="json",
            compliance_level=ComplianceLevel.VERIFIED
        )
        
        # LogiQA Dataset
        datasets["logiqa"] = DatasetMetadata(
            name="LogiQA",
            version="1.0",
            dataset_type=DatasetType.LOGICAL_REASONING,
            license=DatasetLicense(
                license_type=LicenseType.ACADEMIC_ONLY,
                academic_only=True,
                commercial_use=False,
                attribution_required=True,
                citation_required=True,
                paper_reference="Liu, J. et al. (2020). LogiQA: A Challenge Dataset for Machine Reading Comprehension with Logical Reasoning"
            ),
            paper_url="https://arxiv.org/abs/2007.08124",
            description="Logical reasoning questions for machine reading comprehension",
            num_samples=8678,
            size_gb=0.02,
            format="json",
            compliance_level=ComplianceLevel.RESTRICTED
        )
        
        return datasets
    
    async def _register_predefined_datasets(self):
        """Register predefined datasets with the registry"""
        for dataset_id, metadata in self.predefined_datasets.items():
            if dataset_id not in self.registry.datasets:
                await self.registry.register_dataset(dataset_id, metadata)
    
    async def _load_logical_ground_truth(self):
        """Load logical reasoning ground truth data"""
        ground_truth_dir = self.data_directory / "ground_truth"
        if ground_truth_dir.exists():
            for gt_file in ground_truth_dir.glob("*.json"):
                dataset_name = gt_file.stem
                async with aiofiles.open(gt_file, 'r') as f:
                    content = await f.read()
                    gt_data = json.loads(content)
                    
                    ground_truth_items = []
                    for item in gt_data:
                        gt_item = LogicalGroundTruth(
                            task_id=item["task_id"],
                            task_type=item["task_type"],
                            premises=item["premises"],
                            query=item["query"],
                            conclusion=item["conclusion"],
                            reasoning_steps=item["reasoning_steps"],
                            logical_form=item.get("logical_form"),
                            complexity_level=item.get("complexity_level", 1),
                            reasoning_depth=item.get("reasoning_depth", 1),
                            num_entities=item.get("num_entities", 0),
                            num_relations=item.get("num_relations", 0),
                            verified=item.get("verified", False),
                            verification_method=item.get("verification_method", ""),
                            confidence_score=item.get("confidence_score", 1.0)
                        )
                        ground_truth_items.append(gt_item)
                    
                    self.logical_ground_truth[dataset_name] = ground_truth_items
                    self.logger.info(f"Loaded {len(ground_truth_items)} ground truth items for {dataset_name}")
    
    async def download_dataset(self, dataset_id: str, force_redownload: bool = False) -> bool:
        """Download and verify a dataset"""
        if dataset_id not in self.registry.datasets:
            self.logger.error(f"Dataset {dataset_id} not found in registry")
            return False
        
        metadata = self.registry.datasets[dataset_id]
        
        # Check compliance before download
        if metadata.compliance_level not in [ComplianceLevel.VERIFIED, ComplianceLevel.APPROVED]:
            self.logger.error(f"Dataset {dataset_id} not approved for use: {metadata.compliance_level.value}")
            return False
        
        dataset_dir = self.data_directory / "datasets" / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already downloaded
        if not force_redownload and (dataset_dir / "data").exists():
            self.logger.info(f"Dataset {dataset_id} already downloaded")
            return True
        
        if not metadata.download_url:
            self.logger.error(f"No download URL for dataset {dataset_id}")
            return False
        
        try:
            # Download dataset
            self.logger.info(f"Downloading dataset {dataset_id} from {metadata.download_url}")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / "download.zip"
                
                # Download file
                with urlopen(metadata.download_url) as response:
                    with open(temp_path, 'wb') as f:
                        f.write(response.read())
                
                # Extract if zip file
                if metadata.download_url.endswith('.zip'):
                    with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                        zip_ref.extractall(dataset_dir)
                else:
                    # Copy single file
                    import shutil
                    shutil.copy2(temp_path, dataset_dir / "data")
                
                # Verify checksum if available
                if metadata.checksum_sha256:
                    computed_checksum = await self._compute_directory_checksum(dataset_dir)
                    if computed_checksum != metadata.checksum_sha256:
                        self.logger.error(f"Checksum mismatch for dataset {dataset_id}")
                        return False
                
                # Create metadata file
                metadata_file = dataset_dir / "metadata.json"
                async with aiofiles.open(metadata_file, 'w') as f:
                    await f.write(json.dumps({
                        "dataset_id": dataset_id,
                        "name": metadata.name,
                        "version": metadata.version,
                        "download_date": datetime.now(timezone.utc).isoformat(),
                        "compliance_level": metadata.compliance_level.value,
                        "license_type": metadata.license.license_type.value
                    }, indent=2))
                
                self.logger.info(f"Successfully downloaded dataset {dataset_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error downloading dataset {dataset_id}: {e}")
            return False
    
    async def _compute_directory_checksum(self, directory: Path) -> str:
        """Compute SHA256 checksum of directory contents"""
        hasher = hashlib.sha256()
        
        for file_path in sorted(directory.rglob("*")):
            if file_path.is_file():
                async with aiofiles.open(file_path, 'rb') as f:
                    while chunk := await f.read(8192):
                        hasher.update(chunk)
        
        return hasher.hexdigest()
    
    async def validate_license_compliance(self, dataset_id: str, intended_use: str) -> bool:
        """Validate license compliance for intended use"""
        if dataset_id not in self.registry.datasets:
            self.logger.error(f"Dataset {dataset_id} not found")
            return False
        
        metadata = self.registry.datasets[dataset_id]
        
        # Check compliance status
        if metadata.compliance_level not in [ComplianceLevel.VERIFIED, ComplianceLevel.APPROVED]:
            self.logger.warning(f"Dataset {dataset_id} compliance not verified")
            return False
        
        # Check license compatibility
        if not self.registry.check_license_compatibility(dataset_id, intended_use):
            self.logger.error(f"License incompatible for {intended_use} use of dataset {dataset_id}")
            return False
        
        # Check expiry date
        if metadata.license.expiry_date and datetime.now(timezone.utc) > metadata.license.expiry_date:
            self.logger.error(f"Dataset {dataset_id} license expired")
            return False
        
        self.logger.info(f"License compliance validated for dataset {dataset_id}")
        return True
    
    async def create_logical_ground_truth(self, dataset_name: str, 
                                        ground_truth_items: List[LogicalGroundTruth]) -> bool:
        """Create logical reasoning ground truth dataset"""
        try:
            # Validate ground truth items
            for item in ground_truth_items:
                if not self._validate_logical_ground_truth(item):
                    self.logger.error(f"Invalid ground truth item: {item.task_id}")
                    return False
            
            # Save to file
            ground_truth_dir = self.data_directory / "ground_truth"
            ground_truth_dir.mkdir(parents=True, exist_ok=True)
            
            gt_file = ground_truth_dir / f"{dataset_name}.json"
            gt_data = []
            
            for item in ground_truth_items:
                gt_data.append({
                    "task_id": item.task_id,
                    "task_type": item.task_type,
                    "premises": item.premises,
                    "query": item.query,
                    "conclusion": item.conclusion,
                    "reasoning_steps": item.reasoning_steps,
                    "logical_form": item.logical_form,
                    "complexity_level": item.complexity_level,
                    "reasoning_depth": item.reasoning_depth,
                    "num_entities": item.num_entities,
                    "num_relations": item.num_relations,
                    "verified": item.verified,
                    "verification_method": item.verification_method,
                    "confidence_score": item.confidence_score
                })
            
            async with aiofiles.open(gt_file, 'w') as f:
                await f.write(json.dumps(gt_data, indent=2))
            
            self.logical_ground_truth[dataset_name] = ground_truth_items
            self.logger.info(f"Created logical ground truth dataset: {dataset_name} "
                           f"with {len(ground_truth_items)} items")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating logical ground truth: {e}")
            return False
    
    def _validate_logical_ground_truth(self, item: LogicalGroundTruth) -> bool:
        """Validate logical ground truth item"""
        # Basic validation
        if not item.task_id or not item.task_type:
            return False
        
        if not item.premises or not item.query:
            return False
        
        if item.conclusion is None:
            return False
        
        if not item.reasoning_steps:
            return False
        
        # Complexity validation
        if item.complexity_level < 1 or item.complexity_level > 5:
            return False
        
        if item.reasoning_depth < 1:
            return False
        
        # Confidence validation
        if item.confidence_score < 0.0 or item.confidence_score > 1.0:
            return False
        
        return True
    
    async def get_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        total_datasets = len(self.registry.datasets)
        compliant_datasets = len(self.registry.get_compliant_datasets())
        
        license_distribution = {}
        compliance_distribution = {}
        dataset_type_distribution = {}
        
        for metadata in self.registry.datasets.values():
            # License distribution
            license_type = metadata.license.license_type.value
            license_distribution[license_type] = license_distribution.get(license_type, 0) + 1
            
            # Compliance distribution
            compliance_level = metadata.compliance_level.value
            compliance_distribution[compliance_level] = compliance_distribution.get(compliance_level, 0) + 1
            
            # Dataset type distribution
            dataset_type = metadata.dataset_type.value
            dataset_type_distribution[dataset_type] = dataset_type_distribution.get(dataset_type, 0) + 1
        
        # Risk assessment
        high_risk_datasets = [
            dataset_id for dataset_id, metadata in self.registry.datasets.items()
            if metadata.compliance_level in [ComplianceLevel.PENDING, ComplianceLevel.BLOCKED]
            or metadata.license.license_type == LicenseType.UNKNOWN
        ]
        
        return {
            "summary": {
                "total_datasets": total_datasets,
                "compliant_datasets": compliant_datasets,
                "compliance_rate": compliant_datasets / total_datasets if total_datasets > 0 else 0.0,
                "high_risk_datasets": len(high_risk_datasets)
            },
            "distributions": {
                "licenses": license_distribution,
                "compliance_levels": compliance_distribution,
                "dataset_types": dataset_type_distribution
            },
            "risk_assessment": {
                "high_risk_datasets": high_risk_datasets,
                "academic_only_datasets": [
                    dataset_id for dataset_id, metadata in self.registry.datasets.items()
                    if metadata.license.academic_only
                ],
                "expired_licenses": [
                    dataset_id for dataset_id, metadata in self.registry.datasets.items()
                    if metadata.license.expiry_date and 
                    datetime.now(timezone.utc) > metadata.license.expiry_date
                ]
            },
            "ground_truth_coverage": {
                "datasets_with_ground_truth": len(self.logical_ground_truth),
                "total_ground_truth_items": sum(len(items) for items in self.logical_ground_truth.values())
            }
        }
    
    async def update_cognitive_metrics(self):
        """Update cognitive metrics for HOLO-1.5 mesh coordination"""
        report = await self.get_compliance_report()
        
        self.cognitive_metrics["compliance_rate"] = report["summary"]["compliance_rate"]
        self.cognitive_metrics["licensing_risk"] = min(
            report["summary"]["high_risk_datasets"] / max(report["summary"]["total_datasets"], 1), 
            1.0
        )
        
        # Ground truth coverage
        logical_datasets = len([
            d for d in self.registry.datasets.values() 
            if d.dataset_type == DatasetType.LOGICAL_REASONING
        ])
        gt_coverage = len(self.logical_ground_truth) / max(logical_datasets, 1)
        self.cognitive_metrics["ground_truth_coverage"] = min(gt_coverage, 1.0)
        
        # Data integrity (based on verification status)
        verified_datasets = len([
            d for d in self.registry.datasets.values() 
            if d.human_verified and d.checksum_sha256
        ])
        self.cognitive_metrics["data_integrity"] = verified_datasets / max(len(self.registry.datasets), 1)
    
    async def get_cognitive_load(self) -> float:
        """Calculate current cognitive load for HOLO-1.5"""
        await self.update_cognitive_metrics()
        
        # Base load from compliance management
        compliance_load = 1.0 - self.cognitive_metrics["compliance_rate"]
        
        # Additional load from risk factors
        risk_load = self.cognitive_metrics["licensing_risk"] * 0.5
        
        # Load from missing ground truth
        gt_load = (1.0 - self.cognitive_metrics["ground_truth_coverage"]) * 0.3
        
        return min(compliance_load + risk_load + gt_load, 1.0)
    
    async def get_symbolic_depth(self) -> int:
        """Calculate symbolic reasoning depth for HOLO-1.5"""
        # Dataset management has moderate symbolic depth
        # - License compliance checking
        # - Metadata validation
        # - Risk assessment
        return 3
    
    async def generate_trace(self) -> Dict[str, Any]:
        """Generate execution trace for HOLO-1.5"""
        return {
            "component": "DatasetManager",
            "cognitive_metrics": self.cognitive_metrics,
            "compliance_report": await self.get_compliance_report(),
            "registered_datasets": len(self.registry.datasets),
            "ground_truth_datasets": len(self.logical_ground_truth)
        }


# Factory function for easy instantiation
async def create_dataset_manager(data_directory: str) -> DatasetManager:
    """Factory function to create and initialize DatasetManager"""
    manager = DatasetManager(Path(data_directory))
    await manager.async_init()
    return manager


# Export main classes
__all__ = [
    "DatasetManager",
    "DatasetRegistry",
    "DatasetMetadata",
    "DatasetLicense",
    "LogicalGroundTruth",
    "LicenseType",
    "DatasetType", 
    "ComplianceLevel",
    "create_dataset_manager"
]
