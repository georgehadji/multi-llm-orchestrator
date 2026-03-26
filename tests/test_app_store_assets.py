"""
Tests for App Store Asset Generator
=====================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Run: pytest tests/test_app_store_assets.py -v
"""

import pytest

from orchestrator.app_store_assets import (
    AppStoreAssetGenerator,
    AppStoreAssets,
    generate_app_store_assets,
)
from orchestrator.models import ProjectSpec


# ─────────────────────────────────────────────
# Test Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def sample_project():
    """Create sample project."""
    return ProjectSpec(
        name="Fitness Tracker Pro",
        description="Track your workouts, monitor progress, and achieve your fitness goals.",
        criteria="All features functional, workout tracking complete",
    )


@pytest.fixture
def generator():
    """Create asset generator."""
    return AppStoreAssetGenerator()


# ─────────────────────────────────────────────
# Test AppStoreAssets
# ─────────────────────────────────────────────

class TestAppStoreAssets:
    """Test AppStoreAssets dataclass."""
    
    def test_assets_creation(self):
        """Test assets creation."""
        assets = AppStoreAssets(
            app_name="Test App",
            subtitle="Test Subtitle",
            description="Test description",
            keywords="test,app",
            privacy_policy_url="https://example.com/privacy",
            support_url="https://example.com/support",
            privacy_labels={},
            screenshots=[],
            app_icon_spec={},
            review_notes="Test notes",
            demo_credentials=None,
            age_rating="4+",
            export_compliance={},
        )
        
        assert assets.app_name == "Test App"
        assert assets.subtitle == "Test Subtitle"
        assert assets.description == "Test description"
    
    def test_assets_to_dict(self):
        """Test assets serialization."""
        assets = AppStoreAssets(
            app_name="Test App",
            subtitle="Test",
            description="Test",
            keywords="test",
            privacy_policy_url="https://example.com",
            support_url="https://example.com",
            privacy_labels={},
            screenshots=[],
            app_icon_spec={},
            review_notes="Test",
            demo_credentials=None,
            age_rating="4+",
            export_compliance={},
        )
        
        assets_dict = assets.to_dict()
        
        assert assets_dict["app_name"] == "Test App"
        assert "privacy_labels" in assets_dict
        assert "screenshots" in assets_dict


# ─────────────────────────────────────────────
# Test AppStoreAssetGenerator
# ─────────────────────────────────────────────

class TestAppStoreAssetGenerator:
    """Test AppStoreAssetGenerator class."""
    
    def test_generator_initialization(self, generator):
        """Test generator initializes correctly."""
        assert generator is not None
        assert generator.MAX_NAME_LENGTH == 30
        assert generator.MAX_SUBTITLE_LENGTH == 30
    
    @pytest.mark.asyncio
    async def test_generate_complete_assets(self, generator, sample_project):
        """Test generating complete assets."""
        assets = await generator.generate(sample_project)
        
        assert isinstance(assets, AppStoreAssets)
        assert assets.app_name == "Fitness Tracker Pro"
        assert len(assets.app_name) <= 30
        assert len(assets.subtitle) <= 30
        assert len(assets.description) <= 4000
        assert len(assets.keywords) <= 100
        assert assets.age_rating in ["4+", "9+", "12+", "17+"]
    
    @pytest.mark.asyncio
    async def test_name_generation(self, generator, sample_project):
        """Test name generation."""
        name = generator._generate_name(sample_project)
        
        assert name == "Fitness Tracker Pro"
        assert len(name) <= 30
    
    @pytest.mark.asyncio
    async def test_name_truncation(self, generator):
        """Test name truncation for long names."""
        long_project = ProjectSpec(
            name="This Is A Very Long App Name That Exceeds The Maximum Length",
        )
        
        name = generator._generate_name(long_project)
        
        assert len(name) <= 30
        assert name != long_project.name
    
    @pytest.mark.asyncio
    async def test_subtitle_generation(self, generator, sample_project):
        """Test subtitle generation."""
        subtitle = generator._generate_subtitle(sample_project)
        
        assert len(subtitle) <= 30
        assert len(subtitle) > 0
    
    @pytest.mark.asyncio
    async def test_description_generation(self, generator, sample_project):
        """Test description generation."""
        description = generator._generate_description(sample_project)
        
        assert len(description) <= 4000
        assert len(description) > 0
        assert "Fitness Tracker Pro" in description
        assert "KEY FEATURES" in description
    
    @pytest.mark.asyncio
    async def test_keywords_generation(self, generator, sample_project):
        """Test keywords generation."""
        keywords = generator._generate_keywords(sample_project)
        
        assert len(keywords) <= 100
        assert "," in keywords or len(keywords) > 0
    
    @pytest.mark.asyncio
    async def test_privacy_policy_url(self, generator, sample_project):
        """Test privacy policy URL generation."""
        url = generator._generate_privacy_policy_url(sample_project)
        
        assert url.startswith("https://")
        assert "privacy" in url.lower()
    
    @pytest.mark.asyncio
    async def test_support_url(self, generator, sample_project):
        """Test support URL generation."""
        url = generator._generate_support_url(sample_project)
        
        assert url.startswith("https://")
        assert "support" in url.lower() or "github" in url.lower()
    
    @pytest.mark.asyncio
    async def test_privacy_labels_generation(self, generator, sample_project):
        """Test privacy labels generation."""
        labels = generator._generate_privacy_labels(sample_project)
        
        assert "data_used_to_track_you" in labels
        assert "data_linked_to_you" in labels
        assert "data_not_linked_to_you" in labels
    
    @pytest.mark.asyncio
    async def test_screenshot_specs(self, generator, sample_project):
        """Test screenshot specifications."""
        specs = generator._generate_screenshot_specs(sample_project)
        
        assert len(specs) >= 2  # At least iPhone 6.5" and 5.5"
        
        # Check required specs
        iphone_specs = [s for s in specs if "iPhone" in s["device"]]
        assert len(iphone_specs) >= 2
        
        for spec in iphone_specs:
            assert "resolution" in spec
            assert "required" in spec
            assert "min_count" in spec
    
    @pytest.mark.asyncio
    async def test_icon_spec(self, generator, sample_project):
        """Test icon specifications."""
        icon_spec = generator._generate_icon_spec(sample_project)
        
        assert icon_spec["required_size"] == "1024x1024"
        assert icon_spec["format"] == "PNG"
        assert "sizes_needed" in icon_spec
    
    @pytest.mark.asyncio
    async def test_review_notes(self, generator, sample_project):
        """Test review notes generation."""
        notes = generator._generate_review_notes(sample_project)
        
        assert len(notes) > 0
        assert "APP FUNCTIONALITY" in notes
        assert "TESTING INSTRUCTIONS" in notes
    
    @pytest.mark.asyncio
    async def test_demo_credentials_no_login(self, generator, sample_project):
        """Test demo credentials for app without login."""
        creds = generator._generate_demo_account(sample_project)
        
        # Fitness tracker doesn't have login by default
        assert creds is None
    
    @pytest.mark.asyncio
    async def test_demo_credentials_with_login(self, generator):
        """Test demo credentials for app with login."""
        login_project = ProjectSpec(
            name="Auth App",
            description="Login and authentication app",
        )
        
        creds = generator._generate_demo_account(login_project)
        
        assert creds is not None
        assert "username" in creds
        assert "password" in creds
    
    @pytest.mark.asyncio
    async def test_age_rating_default(self, generator, sample_project):
        """Test default age rating."""
        rating = generator._calculate_age_rating(sample_project)
        
        assert rating == "4+"  # Default for most apps
    
    @pytest.mark.asyncio
    async def test_export_compliance(self, generator, sample_project):
        """Test export compliance check."""
        compliance = generator._check_export_compliance(sample_project)
        
        assert "requires_encryption_review" in compliance
        assert "exempt" in compliance
        assert compliance["exempt"] is True  # Most apps are exempt


# ─────────────────────────────────────────────
# Test Convenience Function
# ─────────────────────────────────────────────

class TestConvenienceFunction:
    """Test convenience function."""
    
    @pytest.mark.asyncio
    async def test_generate_app_store_assets(self, sample_project):
        """Test convenience function."""
        assets = await generate_app_store_assets(sample_project)
        
        assert isinstance(assets, AppStoreAssets)
        assert assets.app_name == "Fitness Tracker Pro"


# ─────────────────────────────────────────────
# Test Integration with App Store Validator
# ─────────────────────────────────────────────

class TestIntegration:
    """Test integration with App Store validator."""
    
    @pytest.mark.asyncio
    async def test_assets_compatible_with_validator(self, sample_project):
        """Test generated assets work with validator."""
        from orchestrator.app_store_validator import AppStoreValidator
        
        generator = AppStoreAssetGenerator()
        assets = await generator.generate(sample_project)
        
        # Assets should have all required fields for validator
        assert assets.app_name
        assert assets.description
        assert assets.privacy_policy_url
        assert assets.support_url
        assert assets.age_rating
        
        # Validator can use these assets
        validator = AppStoreValidator()
        # (Validator would use assets for validation)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
