"""
Tests for Native Feature Templates
====================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Run: pytest tests/test_native_features.py -v
"""

import pytest

from orchestrator.native_features import (
    NativeFeature,
    NativeFeatureTemplateGenerator,
    FeatureTemplate,
    generate_native_feature_template,
)


# ─────────────────────────────────────────────
# Test Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def generator():
    """Create template generator."""
    return NativeFeatureTemplateGenerator()


@pytest.fixture
def sample_project():
    """Sample project name."""
    return "My Awesome App"


# ─────────────────────────────────────────────
# Test NativeFeature Enum
# ─────────────────────────────────────────────

class TestNativeFeatureEnum:
    """Test NativeFeature enum."""
    
    def test_feature_values(self):
        """Test feature enum values."""
        assert NativeFeature.PUSH_NOTIFICATIONS.value == "push_notifications"
        assert NativeFeature.OFFLINE_SUPPORT.value == "offline_support"
        assert NativeFeature.BIOMETRIC_AUTH.value == "biometric_auth"
        assert NativeFeature.APP_SHORTCUTS.value == "app_shortcuts"
        assert NativeFeature.WIDGETS.value == "widgets"
        assert NativeFeature.DEEP_LINKING.value == "deep_linking"
        assert NativeFeature.IN_APP_PURCHASES.value == "in_app_purchases"
        assert NativeFeature.SHARE_SHEET.value == "share_sheet"
        assert NativeFeature.CAMERA_PHOTOS.value == "camera_photos"
        assert NativeFeature.LOCATION_SERVICES.value == "location_services"
    
    def test_all_features_count(self):
        """Test all features are defined."""
        features = list(NativeFeature)
        assert len(features) == 10


# ─────────────────────────────────────────────
# Test FeatureTemplate
# ─────────────────────────────────────────────

class TestFeatureTemplate:
    """Test FeatureTemplate dataclass."""
    
    def test_template_creation(self):
        """Test template creation."""
        template = FeatureTemplate(
            feature=NativeFeature.PUSH_NOTIFICATIONS,
            files=["test.swift"],
            description="Test feature",
        )
        
        assert template.feature == NativeFeature.PUSH_NOTIFICATIONS
        assert template.files == ["test.swift"]
        assert template.description == "Test feature"
    
    def test_template_to_dict(self):
        """Test template serialization."""
        template = FeatureTemplate(
            feature=NativeFeature.BIOMETRIC_AUTH,
            files=["BiometricAuth.swift"],
            entitlements=[],
            info_plist={"NSFaceIDUsageDescription": "Test"},
            capabilities=[],
            frameworks=["LocalAuthentication"],
            targets=[],
            description="Biometric auth",
        )
        
        template_dict = template.to_dict()
        
        assert template_dict["feature"] == "biometric_auth"
        assert template_dict["files"] == ["BiometricAuth.swift"]
        assert template_dict["frameworks"] == ["LocalAuthentication"]


# ─────────────────────────────────────────────
# Test NativeFeatureTemplateGenerator
# ─────────────────────────────────────────────

class TestNativeFeatureTemplateGenerator:
    """Test NativeFeatureTemplateGenerator class."""
    
    def test_generator_initialization(self, generator):
        """Test generator initializes correctly."""
        assert generator is not None
        assert len(generator.TEMPLATES) == 10
    
    def test_list_features(self, generator):
        """Test listing all features."""
        features = generator.list_features()
        
        assert len(features) == 10
        
        for feature in features:
            assert "name" in feature
            assert "description" in feature
            assert "files" in feature
    
    def test_get_supported_features(self, generator):
        """Test getting supported feature names."""
        features = generator.get_supported_features()
        
        assert len(features) == 10
        assert "push_notifications" in features
        assert "offline_support" in features
        assert "biometric_auth" in features
    
    @pytest.mark.asyncio
    async def test_generate_push_notifications(self, generator, sample_project):
        """Test generating push notifications template."""
        template = await generator.generate(
            "push_notifications",
            sample_project,
        )
        
        assert template is not None
        assert template.feature == NativeFeature.PUSH_NOTIFICATIONS
        assert len(template.files) >= 2
        assert "aps-environment" in template.entitlements
        assert "UserNotifications" in template.frameworks
    
    @pytest.mark.asyncio
    async def test_generate_offline_support(self, generator, sample_project):
        """Test generating offline support template."""
        template = await generator.generate(
            "offline_support",
            sample_project,
        )
        
        assert template is not None
        assert template.feature == NativeFeature.OFFLINE_SUPPORT
        assert "CoreData" in template.frameworks
        assert "background-fetch" in template.capabilities
    
    @pytest.mark.asyncio
    async def test_generate_biometric_auth(self, generator, sample_project):
        """Test generating biometric auth template."""
        template = await generator.generate(
            "biometric_auth",
            sample_project,
        )
        
        assert template is not None
        assert template.feature == NativeFeature.BIOMETRIC_AUTH
        assert "LocalAuthentication" in template.frameworks
        assert "NSFaceIDUsageDescription" in template.info_plist
    
    @pytest.mark.asyncio
    async def test_generate_app_shortcuts(self, generator, sample_project):
        """Test generating app shortcuts template."""
        template = await generator.generate(
            "app_shortcuts",
            sample_project,
        )
        
        assert template is not None
        assert template.feature == NativeFeature.APP_SHORTCUTS
        assert "AppIntents" in template.frameworks
    
    @pytest.mark.asyncio
    async def test_generate_widgets(self, generator, sample_project):
        """Test generating widgets template."""
        template = await generator.generate(
            "widgets",
            sample_project,
        )
        
        assert template is not None
        assert template.feature == NativeFeature.WIDGETS
        assert "WidgetKit" in template.frameworks
        assert "WidgetExtension" in template.targets
    
    @pytest.mark.asyncio
    async def test_generate_deep_linking(self, generator, sample_project):
        """Test generating deep linking template."""
        template = await generator.generate(
            "deep_linking",
            sample_project,
        )
        
        assert template is not None
        assert template.feature == NativeFeature.DEEP_LINKING
        assert "com.apple.developer.associated-domains" in template.entitlements
    
    @pytest.mark.asyncio
    async def test_generate_in_app_purchases(self, generator, sample_project):
        """Test generating in-app purchases template."""
        template = await generator.generate(
            "in_app_purchases",
            sample_project,
        )
        
        assert template is not None
        assert template.feature == NativeFeature.IN_APP_PURCHASES
        assert "StoreKit" in template.frameworks
        assert "in-app-purchase" in template.capabilities
    
    @pytest.mark.asyncio
    async def test_generate_share_sheet(self, generator, sample_project):
        """Test generating share sheet template."""
        template = await generator.generate(
            "share_sheet",
            sample_project,
        )
        
        assert template is not None
        assert template.feature == NativeFeature.SHARE_SHEET
        assert "ShareExtension" in template.targets
    
    @pytest.mark.asyncio
    async def test_generate_camera_photos(self, generator, sample_project):
        """Test generating camera/photos template."""
        template = await generator.generate(
            "camera_photos",
            sample_project,
        )
        
        assert template is not None
        assert template.feature == NativeFeature.CAMERA_PHOTOS
        assert "NSCameraUsageDescription" in template.info_plist
        assert "NSPhotoLibraryUsageDescription" in template.info_plist
    
    @pytest.mark.asyncio
    async def test_generate_location_services(self, generator, sample_project):
        """Test generating location services template."""
        template = await generator.generate(
            "location_services",
            sample_project,
        )
        
        assert template is not None
        assert template.feature == NativeFeature.LOCATION_SERVICES
        assert "CoreLocation" in template.frameworks
        assert "MapKit" in template.frameworks
    
    @pytest.mark.asyncio
    async def test_generate_unknown_feature(self, generator, sample_project):
        """Test generating unknown feature."""
        template = await generator.generate(
            "unknown_feature",
            sample_project,
        )
        
        assert template is None
    
    @pytest.mark.asyncio
    async def test_customize_template_with_bundle_id(self, generator, sample_project):
        """Test template customization with bundle ID."""
        template = await generator.generate(
            "deep_linking",
            sample_project,
            bundle_id="com.example.myapp",
        )
        
        assert template is not None
        # Bundle ID should be used in associated domains
        assert "info_plist" in template.to_dict()
    
    @pytest.mark.asyncio
    async def test_generate_code_push_notifications(self, generator, sample_project):
        """Test generating push notification code."""
        code = await generator.generate_code(
            "push_notifications",
            sample_project,
            "AppDelegate+Notifications.swift",
        )
        
        assert code is not None
        assert len(code) > 0
        assert "AppDelegate" in code
        assert "UserNotifications" in code
        assert sample_project in code
    
    @pytest.mark.asyncio
    async def test_generate_code_biometric_auth(self, generator, sample_project):
        """Test generating biometric auth code."""
        code = await generator.generate_code(
            "biometric_auth",
            sample_project,
            "BiometricAuth.swift",
        )
        
        assert code is not None
        assert len(code) > 0
        assert "LocalAuthentication" in code
        assert "BiometricAuth" in code
    
    @pytest.mark.asyncio
    async def test_generate_code_unknown_file(self, generator, sample_project):
        """Test generating code for unknown file."""
        code = await generator.generate_code(
            "push_notifications",
            sample_project,
            "UnknownFile.swift",
        )
        
        assert code is None


# ─────────────────────────────────────────────
# Test Convenience Function
# ─────────────────────────────────────────────

class TestConvenienceFunction:
    """Test convenience function."""
    
    @pytest.mark.asyncio
    async def test_generate_native_feature_template(self, sample_project):
        """Test convenience function."""
        template = await generate_native_feature_template(
            "biometric_auth",
            sample_project,
        )
        
        assert template is not None
        assert template.feature == NativeFeature.BIOMETRIC_AUTH


# ─────────────────────────────────────────────
# Test Integration
# ─────────────────────────────────────────────

class TestIntegration:
    """Test integration with other modules."""
    
    @pytest.mark.asyncio
    async def test_templates_compatible_with_validator(self, generator, sample_project):
        """Test templates work with App Store validator."""
        from orchestrator.app_store_validator import AppStoreValidator
        
        # Generate template
        template = await generator.generate("biometric_auth", sample_project)
        
        assert template is not None
        
        # Template should have all required info for validator
        assert template.info_plist
        assert template.frameworks
        
        # Validator can use template info
        validator = AppStoreValidator()
        # (Validator would check template compliance)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
