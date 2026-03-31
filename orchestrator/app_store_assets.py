"""
App Store Asset Generator
==========================
Author: Georgios-Chrysovalantis Chatzivantsidis

Automatic generation of all required App Store submission assets.

Features:
- App metadata (name, subtitle, description, keywords)
- Privacy policy & support URLs
- Privacy labels (data collection)
- Screenshot specifications
- App icon specifications
- Review notes & demo credentials
- Age rating calculation
- Export compliance check

Usage:
    from orchestrator.app_store_assets import AppStoreAssetGenerator

    generator = AppStoreAssetGenerator()
    assets = await generator.generate(project_spec)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import ProjectSpec

logger = logging.getLogger("orchestrator.app_store_assets")


@dataclass
class AppStoreAssets:
    """
    Complete App Store submission assets.

    Attributes:
        app_name: App name (max 30 chars)
        subtitle: App subtitle (max 30 chars)
        description: Full description (max 4000 chars)
        keywords: Comma-separated keywords (max 100 chars)
        privacy_policy_url: Privacy policy URL
        support_url: Support URL
        privacy_labels: Data collection privacy labels
        screenshots: Screenshot specifications
        app_icon_spec: App icon specifications
        review_notes: Notes for App Review team
        demo_credentials: Demo account credentials
        age_rating: Age rating (4+, 9+, 12+, 17+)
        export_compliance: Export compliance info
    """

    app_name: str
    subtitle: str
    description: str
    keywords: str
    privacy_policy_url: str
    support_url: str
    privacy_labels: dict[str, Any]
    screenshots: list[dict[str, Any]]
    app_icon_spec: dict[str, Any]
    review_notes: str
    demo_credentials: dict[str, str] | None
    age_rating: str
    export_compliance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "app_name": self.app_name,
            "subtitle": self.subtitle,
            "description": self.description,
            "keywords": self.keywords,
            "privacy_policy_url": self.privacy_policy_url,
            "support_url": self.support_url,
            "privacy_labels": self.privacy_labels,
            "screenshots": self.screenshots,
            "app_icon_spec": self.app_icon_spec,
            "review_notes": self.review_notes,
            "demo_credentials": self.demo_credentials,
            "age_rating": self.age_rating,
            "export_compliance": self.export_compliance,
        }


class AppStoreAssetGenerator:
    """
    Generate all required App Store submission assets.

    Usage:
        generator = AppStoreAssetGenerator()
        assets = await generator.generate(project_spec)
    """

    # App name constraints
    MAX_NAME_LENGTH = 30
    MIN_NAME_LENGTH = 2  # FIX-003a: App Store requires at least 2 characters
    MAX_SUBTITLE_LENGTH = 30
    MAX_KEYWORDS_LENGTH = 100
    MAX_DESCRIPTION_LENGTH = 4000

    # Age ratings
    AGE_RATINGS = ["4+", "9+", "12+", "17+"]

    def __init__(self):
        """Initialize asset generator."""
        pass

    async def generate(self, project: ProjectSpec) -> AppStoreAssets:
        """
        Generate all App Store assets for a project.

        Args:
            project: Project specification

        Returns:
            Complete AppStoreAssets
        """
        logger.info(f"Generating App Store assets for: {project.name}")

        return AppStoreAssets(
            # Required metadata
            app_name=self._generate_name(project),
            subtitle=self._generate_subtitle(project),
            description=self._generate_description(project),
            keywords=self._generate_keywords(project),
            privacy_policy_url=self._generate_privacy_policy_url(project),
            support_url=self._generate_support_url(project),
            # Privacy labels
            privacy_labels=self._generate_privacy_labels(project),
            # Visual assets
            screenshots=self._generate_screenshot_specs(project),
            app_icon_spec=self._generate_icon_spec(project),
            # Review notes
            review_notes=self._generate_review_notes(project),
            demo_credentials=self._generate_demo_account(project),
            # Compliance
            age_rating=self._calculate_age_rating(project),
            export_compliance=self._check_export_compliance(project),
        )

    def _generate_name(self, project: ProjectSpec) -> str:
        """
        Generate app name (max 30 characters).

        Args:
            project: Project specification

        Returns:
            App name
        """
        # FIX-003a: Validate project name before generating app name
        raw_name = getattr(project, "name", "").strip()

        # Handle empty or whitespace-only names
        if not raw_name or len(raw_name) < self.MIN_NAME_LENGTH:
            # Generate fallback name from description or use generic
            if hasattr(project, "description") and project.description:
                # Extract first few meaningful words from description
                words = project.description.split()[:3]
                raw_name = " ".join(words) if words else "MyApp"
            else:
                raw_name = "MyApp"

        # Use project name, truncated if needed
        name = raw_name[: self.MAX_NAME_LENGTH].strip()

        # Ensure it doesn't end with space
        if len(name) < len(raw_name):
            name = name.rsplit(" ", 1)[0]

        logger.debug(f"Generated app name: {name}")
        return name

    def _generate_subtitle(self, project: ProjectSpec) -> str:
        """
        Generate app subtitle (max 30 characters).

        Args:
            project: Project specification

        Returns:
            App subtitle
        """
        # Extract key value proposition from description
        if hasattr(project, "description") and project.description:
            # Take first sentence, truncate if needed
            first_sentence = project.description.split(".")[0]
            subtitle = first_sentence[: self.MAX_SUBTITLE_LENGTH].strip()
        else:
            # Generic subtitle based on project type
            subtitle = f"Professional {project.name} App"

        subtitle = subtitle[: self.MAX_SUBTITLE_LENGTH].strip()
        logger.debug(f"Generated subtitle: {subtitle}")
        return subtitle

    def _generate_description(self, project: ProjectSpec) -> str:
        """
        Generate full description (max 4000 characters).

        Args:
            project: Project specification

        Returns:
            App description
        """
        # Build structured description
        description_parts = []

        # Opening hook
        description_parts.append(
            f"Welcome to {project.name} - your ultimate solution for {self._extract_category(project)}."
        )
        description_parts.append("")

        # Key features
        description_parts.append("✨ KEY FEATURES:")
        description_parts.append(self._generate_features_list(project))
        description_parts.append("")

        # Description from project
        if hasattr(project, "description") and project.description:
            description_parts.append("📖 ABOUT:")
            description_parts.append(project.description)
            description_parts.append("")

        # What's new
        description_parts.append("🆕 WHAT'S NEW:")
        description_parts.append("- Initial release")
        description_parts.append("- All features fully functional")
        description_parts.append("- Optimized for iOS 17+")
        description_parts.append("")

        # Call to action
        description_parts.append("📲 Download now and experience the difference!")

        description = "\n".join(description_parts)

        # Truncate if needed
        if len(description) > self.MAX_DESCRIPTION_LENGTH:
            description = description[: self.MAX_DESCRIPTION_LENGTH - 3] + "..."

        logger.debug(f"Generated description ({len(description)} chars)")
        return description

    def _generate_keywords(self, project: ProjectSpec) -> str:
        """
        Generate keywords (max 100 characters, comma-separated).

        Args:
            project: Project specification

        Returns:
            Comma-separated keywords
        """
        # Extract relevant keywords from project
        base_keywords = []

        # Project name variations
        name_words = project.name.lower().split()
        base_keywords.extend([w for w in name_words if len(w) > 3])

        # Category-based keywords
        category = self._extract_category(project)
        base_keywords.append(category)

        # Common app keywords
        base_keywords.extend(["app", "tool", "utility", "professional"])

        # Remove duplicates and join
        unique_keywords = list(dict.fromkeys(base_keywords))
        keywords = ",".join(unique_keywords)

        # Truncate if needed
        if len(keywords) > self.MAX_KEYWORDS_LENGTH:
            keywords = keywords[: self.MAX_KEYWORDS_LENGTH]
            # Ensure we don't cut in middle of word
            keywords = keywords.rsplit(",", 1)[0]

        logger.debug(f"Generated keywords: {keywords}")
        return keywords

    def _generate_privacy_policy_url(self, project: ProjectSpec) -> str:
        """
        Generate privacy policy URL.

        Args:
            project: Project specification

        Returns:
            Privacy policy URL
        """
        # Default to hosted privacy policy generator
        # In production, this should be actual URL
        return "https://www.privacypolicies.com/live/your-app-id"

    def _generate_support_url(self, project: ProjectSpec) -> str:
        """
        Generate support URL.

        Args:
            project: Project specification

        Returns:
            Support URL
        """
        # Default to GitHub issues or support page
        return "https://github.com/yourusername/yourapp/issues"

    def _generate_privacy_labels(self, project: ProjectSpec) -> dict[str, Any]:
        """
        Generate App Store privacy labels.

        Args:
            project: Project specification

        Returns:
            Privacy labels dictionary
        """
        # Default: minimal data collection
        privacy_labels = {
            "data_used_to_track_you": [],
            "data_linked_to_you": [],
            "data_not_linked_to_you": [],
        }

        # Check if app has specific features that require data
        if self._has_login_feature(project):
            privacy_labels["data_linked_to_you"].append("Contact Info")
            privacy_labels["data_linked_to_you"].append("User Content")

        if self._has_analytics(project):
            privacy_labels["data_used_to_track_you"].append("Usage Data")
            privacy_labels["data_used_to_track_you"].append("Identifiers")

        if self._has_location_feature(project):
            privacy_labels["data_linked_to_you"].append("Location")

        logger.debug(f"Generated privacy labels: {privacy_labels}")
        return privacy_labels

    def _generate_screenshot_specs(self, project: ProjectSpec) -> list[dict[str, Any]]:
        """
        Generate screenshot specifications.

        Args:
            project: Project specification

        Returns:
            List of screenshot specs
        """
        specs = []

        # iPhone 6.5" display (required)
        specs.append(
            {
                "device": 'iPhone 6.5"',
                "resolution": "1284x2778",
                "orientation": "portrait",
                "required": True,
                "min_count": 1,
                "description": "Main screen showing primary functionality",
            }
        )

        # iPhone 5.5" display (required)
        specs.append(
            {
                "device": 'iPhone 5.5"',
                "resolution": "1242x2208",
                "orientation": "portrait",
                "required": True,
                "min_count": 1,
                "description": "Main screen showing primary functionality",
            }
        )

        # iPad Pro 12.9" (optional but recommended)
        specs.append(
            {
                "device": 'iPad Pro 12.9"',
                "resolution": "2048x2732",
                "orientation": "portrait",
                "required": False,
                "min_count": 0,
                "description": "iPad optimized interface",
            }
        )

        logger.debug(f"Generated screenshot specs for {len(specs)} devices")
        return specs

    def _generate_icon_spec(self, project: ProjectSpec) -> dict[str, Any]:
        """
        Generate app icon specifications.

        Args:
            project: Project specification

        Returns:
            Icon specifications
        """
        return {
            "required_size": "1024x1024",
            "format": "PNG",
            "color_space": "sRGB",
            "layers": "Flattened",
            "transparency": "Not allowed",
            "corner_radius": "Apple adds automatically",
            "sizes_needed": [
                "1024x1024 (App Store)",
                "180x180 (iPhone @3x)",
                "120x120 (iPhone @2x)",
                "152x152 (iPad @2x)",
                "167x167 (iPad Pro @2x)",
                "128x128 (Mac)",
                "256x256 (Mac)",
                "512x512 (Mac)",
            ],
        }

    def _generate_review_notes(self, project: ProjectSpec) -> str:
        """
        Generate notes for App Review team.

        Args:
            project: Project specification

        Returns:
            Review notes
        """
        notes = []
        notes.append("Thank you for reviewing our app!")
        notes.append("")
        notes.append("APP FUNCTIONALITY:")
        notes.append(f"This app provides {self._extract_category(project)} functionality.")
        notes.append("")
        notes.append("TESTING INSTRUCTIONS:")
        notes.append("1. All features are fully functional")
        notes.append("2. No special hardware required")
        notes.append("3. Works offline with limited functionality")
        notes.append("")

        if self._has_login_feature(project):
            notes.append("DEMO ACCOUNT:")
            notes.append("Username: demo@example.com")
            notes.append("Password: DemoPass123!")
            notes.append("")

        notes.append("COMPLIANCE:")
        notes.append("- No user data sold to third parties")
        notes.append("- All data encrypted in transit")
        notes.append("- Privacy policy included")

        review_notes = "\n".join(notes)
        logger.debug(f"Generated review notes ({len(review_notes)} chars)")
        return review_notes

    def _generate_demo_account(self, project: ProjectSpec) -> dict[str, str] | None:
        """
        Generate demo account credentials.

        Args:
            project: Project specification

        Returns:
            Demo credentials or None
        """
        if self._has_login_feature(project):
            return {
                "username": "demo@example.com",
                "password": "DemoPass123!",
                "notes": "Full access to all features",
            }
        return None

    def _calculate_age_rating(self, project: ProjectSpec) -> str:
        """
        Calculate age rating based on app content.

        Args:
            project: Project specification

        Returns:
            Age rating (4+, 9+, 12+, 17+)
        """
        # Default to 4+ for most apps
        rating = "4+"

        # Check for mature content
        if self._has_mature_content(project):
            rating = "17+"
        elif self._has_mild_content(project):
            rating = "12+"
        elif self._has_cartoon_violence(project):
            rating = "9+"

        logger.debug(f"Calculated age rating: {rating}")
        return rating

    def _check_export_compliance(self, project: ProjectSpec) -> dict[str, Any]:
        """
        Check export compliance requirements.

        Args:
            project: Project specification

        Returns:
            Export compliance info
        """
        # Most apps qualify for exemption
        compliance = {
            "requires_encryption_review": False,
            "exempt": True,
            "exemption_reason": "App uses standard encryption available to all apps",
            "ccats_number": None,  # Only if not exempt
        }

        # Check if app uses custom encryption
        if self._uses_custom_encryption(project):
            compliance["requires_encryption_review"] = True
            compliance["exempt"] = False

        logger.debug(f"Export compliance: {compliance}")
        return compliance

    # ─────────────────────────────────────────────
    # Helper Methods
    # ─────────────────────────────────────────────

    def _extract_category(self, project: ProjectSpec) -> str:
        """Extract app category from project."""
        # Simple heuristic based on project name/description
        name_lower = project.name.lower()

        if any(word in name_lower for word in ["fitness", "health", "workout"]):
            return "Health & Fitness"
        elif any(word in name_lower for word in ["game", "puzzle", "quiz"]):
            return "Games"
        elif any(word in name_lower for word in ["photo", "image", "camera"]):
            return "Photo & Video"
        elif any(word in name_lower for word in ["music", "audio", "sound"]):
            return "Music"
        elif any(word in name_lower for word in ["news", "magazine", "journal"]):
            return "News"
        elif any(word in name_lower for word in ["shop", "store", "buy"]):
            return "Shopping"
        else:
            return "Utilities"

    def _generate_features_list(self, project: ProjectSpec) -> str:
        """Generate features list for description."""
        features = [
            "• Intuitive and user-friendly interface",
            "• Fast and responsive performance",
            "• Offline support for core features",
            "• Regular updates with new features",
            "• Secure data handling",
        ]
        return "\n".join(features)

    def _has_login_feature(self, project: ProjectSpec) -> bool:
        """Check if app has login/authentication."""
        name_lower = project.name.lower()
        return any(word in name_lower for word in ["auth", "login", "account", "member"])

    def _has_analytics(self, project: ProjectSpec) -> bool:
        """Check if app has analytics."""
        # Default to true for most apps
        return True

    def _has_location_feature(self, project: ProjectSpec) -> bool:
        """Check if app uses location."""
        name_lower = project.name.lower()
        return any(word in name_lower for word in ["map", "location", "gps", "nearby"])

    def _has_mature_content(self, project: ProjectSpec) -> bool:
        """Check for mature content."""
        # Most business apps don't have mature content
        return False

    def _has_mild_content(self, project: ProjectSpec) -> bool:
        """Check for mild content."""
        return False

    def _has_cartoon_violence(self, project: ProjectSpec) -> bool:
        """Check for cartoon violence."""
        return False

    def _uses_custom_encryption(self, project: ProjectSpec) -> bool:
        """Check if app uses custom encryption."""
        # Most apps use standard TLS, which is exempt
        return False


# ─────────────────────────────────────────────
# Convenience Function
# ─────────────────────────────────────────────


async def generate_app_store_assets(project: ProjectSpec) -> AppStoreAssets:
    """
    Convenience function to generate App Store assets.

    Args:
        project: Project specification

    Returns:
        AppStoreAssets
    """
    generator = AppStoreAssetGenerator()
    return await generator.generate(project)
