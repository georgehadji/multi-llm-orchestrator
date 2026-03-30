"""
Native Feature Integration Templates
=====================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Pre-built templates for native iOS features that help apps pass
the App Store "minimum functionality" threshold.

Features:
- Push Notifications
- Offline Support (CoreData)
- Biometric Authentication (FaceID/TouchID)
- App Shortcuts (App Intents)
- Widgets (WidgetKit)
- Deep Linking
- In-App Purchases
- Share Sheet
- Camera/Photo Library
- Location Services

Usage:
    from orchestrator.native_features import NativeFeatureTemplateGenerator

    generator = NativeFeatureTemplateGenerator()
    templates = await generator.generate("push_notifications", project_name)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("orchestrator.native_features")


class NativeFeature(str, Enum):
    """Supported native iOS features."""
    PUSH_NOTIFICATIONS = "push_notifications"
    OFFLINE_SUPPORT = "offline_support"
    BIOMETRIC_AUTH = "biometric_auth"
    APP_SHORTCUTS = "app_shortcuts"
    WIDGETS = "widgets"
    DEEP_LINKING = "deep_linking"
    IN_APP_PURCHASES = "in_app_purchases"
    SHARE_SHEET = "share_sheet"
    CAMERA_PHOTOS = "camera_photos"
    LOCATION_SERVICES = "location_services"


@dataclass
class FeatureTemplate:
    """
    Template for a native iOS feature.

    Attributes:
        feature: Feature type
        files: List of Swift files to generate
        entitlements: Required entitlements
        info_plist: Required Info.plist entries
        capabilities: Required capabilities
        frameworks: Required frameworks
        targets: Additional targets (extensions, etc.)
        description: Feature description
    """
    feature: NativeFeature
    files: list[str]
    entitlements: list[str] = field(default_factory=list)
    info_plist: dict[str, str] = field(default_factory=dict)
    capabilities: list[str] = field(default_factory=list)
    frameworks: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature": self.feature.value,
            "files": self.files,
            "entitlements": self.entitlements,
            "info_plist": self.info_plist,
            "capabilities": self.capabilities,
            "frameworks": self.frameworks,
            "targets": self.targets,
            "description": self.description,
        }


class NativeFeatureTemplateGenerator:
    """
    Generate native iOS feature templates.

    Usage:
        generator = NativeFeatureTemplateGenerator()
        template = await generator.generate("push_notifications", "MyApp")
    """

    # Feature templates
    TEMPLATES: dict[NativeFeature, FeatureTemplate] = {
        NativeFeature.PUSH_NOTIFICATIONS: FeatureTemplate(
            feature=NativeFeature.PUSH_NOTIFICATIONS,
            files=[
                "AppDelegate+Notifications.swift",
                "NotificationService.swift",
                "NotificationManager.swift",
            ],
            entitlements=["aps-environment"],
            info_plist={
                "NSUserNotificationsUsageDescription": "We need notification permission to send you important updates about your activity.",
            },
            capabilities=["push-notifications"],
            frameworks=["UserNotifications"],
            targets=["NotificationServiceExtension"],
            description="Push notifications for user engagement and retention",
        ),

        NativeFeature.OFFLINE_SUPPORT: FeatureTemplate(
            feature=NativeFeature.OFFLINE_SUPPORT,
            files=[
                "OfflineManager.swift",
                "CoreDataStack.swift",
                "DataSyncManager.swift",
                "Models+CoreDataClass.swift",
            ],
            entitlements=[],
            info_plist={
                "UIBackgroundModes": ["fetch"],
            },
            capabilities=["background-fetch"],
            frameworks=["CoreData"],
            targets=[],
            description="Offline data support with CoreData persistence and background sync",
        ),

        NativeFeature.BIOMETRIC_AUTH: FeatureTemplate(
            feature=NativeFeature.BIOMETRIC_AUTH,
            files=[
                "BiometricAuth.swift",
                "BiometricAuthViewModel.swift",
            ],
            entitlements=[],
            info_plist={
                "NSFaceIDUsageDescription": "FaceID is used to securely authenticate you and protect your personal data.",
            },
            capabilities=[],
            frameworks=["LocalAuthentication"],
            targets=[],
            description="FaceID/TouchID biometric authentication for enhanced security",
        ),

        NativeFeature.APP_SHORTCUTS: FeatureTemplate(
            feature=NativeFeature.APP_SHORTCUTS,
            files=[
                "AppIntents.swift",
                "AppShortcuts.swift",
                "IntentHandler.swift",
            ],
            entitlements=[],
            info_plist={},
            capabilities=[],
            frameworks=["AppIntents", "Intents"],
            targets=[],
            description="App Intents and shortcuts for Siri integration and quick actions",
        ),

        NativeFeature.WIDGETS: FeatureTemplate(
            feature=NativeFeature.WIDGETS,
            files=[
                "Widget.swift",
                "WidgetBundle.swift",
                "WidgetIntent.swift",
                "WidgetViews.swift",
            ],
            entitlements=[],
            info_plist={},
            capabilities=[],
            frameworks=["WidgetKit"],
            targets=["WidgetExtension"],
            description="Home screen widgets for quick access to key information",
        ),

        NativeFeature.DEEP_LINKING: FeatureTemplate(
            feature=NativeFeature.DEEP_LINKING,
            files=[
                "DeepLinkManager.swift",
                "URLHandler.swift",
                "UniversalLinks.swift",
            ],
            entitlements=["com.apple.developer.associated-domains"],
            info_plist={
                "CFBundleURLTypes": [
                    {
                        "CFBundleURLName": "$(PRODUCT_NAME)",
                        "CFBundleURLSchemes": ["myapp"],
                    }
                ],
            },
            capabilities=["associated-domains"],
            frameworks=[],
            targets=[],
            description="Deep linking and universal links for seamless user experience",
        ),

        NativeFeature.IN_APP_PURCHASES: FeatureTemplate(
            feature=NativeFeature.IN_APP_PURCHASES,
            files=[
                "StoreManager.swift",
                "PurchaseManager.swift",
                "ProductView.swift",
                "SubscriptionView.swift",
            ],
            entitlements=["in-app-purchase"],
            info_plist={},
            capabilities=["in-app-purchase"],
            frameworks=["StoreKit"],
            targets=[],
            description="In-app purchases and subscriptions for monetization",
        ),

        NativeFeature.SHARE_SHEET: FeatureTemplate(
            feature=NativeFeature.SHARE_SHEET,
            files=[
                "ShareSheet.swift",
                "ActivityItemProvider.swift",
                "ShareViewController.swift",
            ],
            entitlements=[],
            info_plist={},
            capabilities=[],
            frameworks=["UIKit"],
            targets=["ShareExtension"],
            description="Share sheet extension for content sharing",
        ),

        NativeFeature.CAMERA_PHOTOS: FeatureTemplate(
            feature=NativeFeature.CAMERA_PHOTOS,
            files=[
                "CameraManager.swift",
                "PhotoPicker.swift",
                "ImageProcessor.swift",
            ],
            entitlements=[],
            info_plist={
                "NSCameraUsageDescription": "Camera access is used to capture photos for your profile and content.",
                "NSPhotoLibraryUsageDescription": "Photo library access is used to select and save images.",
                "NSPhotoLibraryAddUsageDescription": "Photo library write access is used to save your edited images.",
            },
            capabilities=[],
            frameworks=["Photos", "AVFoundation"],
            targets=[],
            description="Camera and photo library integration for media capture",
        ),

        NativeFeature.LOCATION_SERVICES: FeatureTemplate(
            feature=NativeFeature.LOCATION_SERVICES,
            files=[
                "LocationManager.swift",
                "LocationViewModel.swift",
                "MapView.swift",
            ],
            entitlements=[],
            info_plist={
                "NSLocationWhenInUseUsageDescription": "Location access is used to show nearby places and provide location-based features.",
                "NSLocationAlwaysAndWhenInUseUsageDescription": "Location access is used for background tracking and location-based notifications.",
            },
            capabilities=[],
            frameworks=["CoreLocation", "MapKit"],
            targets=[],
            description="Location services for maps and location-based features",
        ),
    }

    def __init__(self):
        """Initialize template generator."""
        pass

    async def generate(
        self,
        feature: str,
        project_name: str,
        bundle_id: str | None = None,
    ) -> FeatureTemplate | None:
        """
        Generate template for a native feature.

        Args:
            feature: Feature name (e.g., "push_notifications")
            project_name: Project/app name
            bundle_id: Bundle identifier

        Returns:
            FeatureTemplate or None if feature not found
        """
        logger.info(f"Generating template for: {feature}")

        # Find feature
        feature_enum = self._get_feature_enum(feature)
        if feature_enum is None:
            logger.warning(f"Unknown feature: {feature}")
            return None

        template = self.TEMPLATES.get(feature_enum)
        if template is None:
            logger.warning(f"No template for: {feature}")
            return None

        # Customize template for project
        customized = self._customize_template(template, project_name, bundle_id)

        logger.info(f"Generated template for {feature} with {len(template.files)} files")
        return customized

    def list_features(self) -> list[dict[str, Any]]:
        """
        List all available features.

        Returns:
            List of feature info dictionaries
        """
        return [
            {
                "name": feature.value,
                "description": template.description,
                "files": template.files,
                "frameworks": template.frameworks,
                "capabilities": template.capabilities,
            }
            for feature, template in self.TEMPLATES.items()
        ]

    def get_supported_features(self) -> list[str]:
        """
        Get list of supported feature names.

        Returns:
            List of feature names
        """
        return [f.value for f in NativeFeature]

    def _get_feature_enum(self, feature: str) -> NativeFeature | None:
        """Get feature enum from string."""
        try:
            return NativeFeature(feature)
        except ValueError:
            return None

    def _customize_template(
        self,
        template: FeatureTemplate,
        project_name: str,
        bundle_id: str | None,
    ) -> FeatureTemplate:
        """
        Customize template for specific project.

        Args:
            template: Original template
            project_name: Project name
            bundle_id: Bundle identifier

        Returns:
            Customized template
        """
        # Create a copy to avoid modifying original
        customized = FeatureTemplate(
            feature=template.feature,
            files=template.files.copy(),
            entitlements=template.entitlements.copy(),
            info_plist=dict(template.info_plist),
            capabilities=template.capabilities.copy(),
            frameworks=template.frameworks.copy(),
            targets=template.targets.copy(),
            description=template.description,
        )

        # Customize Info.plist entries
        for key, value in customized.info_plist.items():
            if isinstance(value, str):
                # Replace placeholders
                customized.info_plist[key] = value.replace(
                    "$(PRODUCT_NAME)", project_name
                )

        # Add bundle ID to entitlements if needed
        if bundle_id and "com.apple.developer.associated-domains" in customized.entitlements:
            customized.info_plist["NSAssociatedDomains"] = [
                f"applinks:{bundle_id.replace('.', '-')}.com"
            ]

        return customized

    async def generate_code(
        self,
        feature: str,
        project_name: str,
        file_name: str,
    ) -> str | None:
        """
        Generate actual Swift code for a feature file.

        Args:
            feature: Feature name
            project_name: Project name
            file_name: Specific file to generate

        Returns:
            Swift code or None
        """
        template = await self.generate(feature, project_name)
        if template is None:
            return None

        if file_name not in template.files:
            logger.warning(f"File {file_name} not in template")
            return None

        # Generate code based on file type
        generator = self._get_code_generator(feature, file_name)
        if generator is None:
            return None

        return generator(project_name)

    def _get_code_generator(
        self,
        feature: str,
        file_name: str,
    ):
        """Get code generator function for specific file."""
        generators = {
            "push_notifications": self._generate_push_notification_code,
            "offline_support": self._generate_offline_support_code,
            "biometric_auth": self._generate_biometric_auth_code,
            "app_shortcuts": self._generate_app_shortcuts_code,
            "widgets": self._generate_widget_code,
        }

        return generators.get(feature)

    # ─────────────────────────────────────────────
    # Code Generators for Specific Features
    # ─────────────────────────────────────────────

    def _generate_push_notification_code(
        self,
        project_name: str,
    ) -> str:
        """Generate push notification code."""
        project_name.replace(" ", "")

        return f'''//
//  AppDelegate+Notifications.swift
//  {project_name}
//
//  Generated by AI Orchestrator
//

import UIKit
import UserNotifications

extension AppDelegate {{

    func setupNotifications() {{
        UNUserNotificationCenter.current().delegate = self

        // Request authorization
        UNUserNotificationCenter.current().requestAuthorization(
            options: [.alert, .badge, .sound]
        ) {{ granted, error in
            if granted {{
                print("Notification permission granted")
                DispatchQueue.main.async {{
                    UIApplication.shared.registerForRemoteNotifications()
                }}
            }} else if let error = error {{
                print("Notification error: {{error.localizedDescription}}")
            }}
        }}
    }}
}}

// MARK: - UNUserNotificationCenterDelegate

extension AppDelegate: UNUserNotificationCenterDelegate {{

    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification,
        withCompletionHandler completionHandler:
        @escaping (UNNotificationPresentationOptions) -> Void
    ) {{
        // Show notification even when app is in foreground
        completionHandler([.banner, .badge, .sound])
    }}

    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        didReceive response: UNNotificationResponse,
        withCompletionHandler completionHandler: @escaping () -> Void
    ) {{
        // Handle notification tap
        let userInfo = response.notification.request.content.userInfo
        print("Notification tapped: {{userInfo}}")
        completionHandler()
    }}
}}

// MARK: - Remote Notifications

extension AppDelegate {{

    func application(
        _ application: UIApplication,
        didRegisterForRemoteNotificationsWithDeviceToken deviceToken: Data
    ) {{
        let token = deviceToken.map {{ String(format: "%02.2hhx", $0) }}.joined()
        print("Device token: {{token}}")
        // Send token to your server
    }}

    func application(
        _ application: UIApplication,
        didFailToRegisterForRemoteNotificationsWithError error: Error
    ) {{
        print("Failed to register: {{error.localizedDescription}}")
    }}
}}
'''

    def _generate_biometric_auth_code(
        self,
        project_name: str,
    ) -> str:
        """Generate biometric authentication code."""
        return f'''//
//  BiometricAuth.swift
//  {project_name}
//
//  Generated by AI Orchestrator
//

import LocalAuthentication
import Foundation

enum BiometricAuthError: LocalizedError {{
    case notAvailable
    case notEnrolled
    case lockout
    case userCancel
    case unknown

    var errorDescription: String? {{
        switch self {{
        case .notAvailable:
            return "Biometric authentication is not available on this device."
        case .notEnrolled:
            return "No biometric credentials enrolled."
        case .lockout:
            return "Too many failed attempts. Please use passcode."
        case .userCancel:
            return "Authentication cancelled by user."
        case .unknown:
            return "An unknown error occurred."
        }}
    }}
}}

class BiometricAuth {{

    static let shared = BiometricAuth()

    private let context = LAContext()

    private init() {{}}

    var isAvailable: Bool {{
        return context.canEvaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, error: nil)
    }}

    var biometryType: LABiometryType {{
        return context.biometryType
    }}

    func authenticate(reason: String = "Authenticate to access your account") async throws {{
        guard isAvailable else {{
            throw BiometricAuthError.notAvailable
        }}

        do {{
            let success = try await context.evaluatePolicy(
                .deviceOwnerAuthenticationWithBiometrics,
                localizedReason: reason
            )

            guard success else {{
                throw BiometricAuthError.userCancel
            }}
        }} catch {{
            throw mapError(error)
        }}
    }}

    private func mapError(_ error: Error) -> BiometricAuthError {{
        guard let laError = error as? LAError else {{
            return .unknown
        }}

        switch laError.code {{
        case .authenticationFailed:
            return .notEnrolled
        case .userCancel:
            return .userCancel
        case .lockout:
            return .lockout
        default:
            return .unknown
        }}
    }}
}}
'''

    def _generate_offline_support_code(
        self,
        project_name: str,
    ) -> str:
        """Generate offline support code."""
        return f'''//
//  OfflineManager.swift
//  {project_name}
//
//  Generated by AI Orchestrator
//

import Foundation
import CoreData

class OfflineManager {{

    static let shared = OfflineManager()

    private let coreDataStack: CoreDataStack

    init(coreDataStack: CoreDataStack = .shared) {{
        self.coreDataStack = coreDataStack
    }}

    var isOnline: Bool {{
        // Implement network reachability check
        return true
    }}

    // MARK: - Data Sync

    func syncData() async throws {{
        guard isOnline else {{
            print("Offline mode - data will sync when online")
            return
        }}

        let context = coreDataStack.backgroundContext

        try await context.perform {{
            // Fetch pending sync items
            let request = NSFetchRequest<NSManagedObject>(entityName: "SyncItem")
            request.predicate = NSPredicate(format: "isSynced == NO")

            let items = try context.fetch(request)

            for item in items {{
                // Sync each item
                try self.syncItem(item)
            }}
        }}
    }}

    private func syncItem(_ item: NSManagedObject) throws {{
        // Implement sync logic
        item.setValue(true, forKey: "isSynced")
        try item.managedObjectContext?.save()
    }}

    // MARK: - Cache

    func cacheData<T: Codable>(_ data: T, forKey key: String) throws {{
        let encoder = JSONEncoder()
        let encoded = try encoder.encode(data)

        UserDefaults.standard.set(encoded, forKey: key)
    }}

    func cachedData<T: Codable>(forKey key: String, as type: T.Type) -> T? {{
        guard let data = UserDefaults.standard.data(forKey: key) else {{
            return nil
        }}

        let decoder = JSONDecoder()
        return try? decoder.decode(T.self, from: data)
    }}
}}
'''

    def _generate_app_shortcuts_code(
        self,
        project_name: str,
    ) -> str:
        """Generate app shortcuts code."""
        return f'''//
//  AppIntents.swift
//  {project_name}
//
//  Generated by AI Orchestrator
//

import AppIntents

struct OpenAppIntent: AppIntent {{
    static var title: LocalizedStringResource = "Open {project_name}"
    static var description = IntentDescription("Open the app to a specific screen")
    static var openAppWhenRun = true

    @Parameter(title: "Screen")
    var screenName: String

    init() {{}}

    func perform() async throws -> some IntentResult {{
        // Handle opening to specific screen
        NotificationCenter.default.post(
            name: NSNotification.Name("OpenScreen"),
            object: nil,
            userInfo: ["screenName": screenName]
        )
        return .result()
    }}
}}

struct QuickActionIntent: AppIntent {{
    static var title: LocalizedStringResource = "Quick Action"
    static var description = IntentDescription("Perform a quick action from home screen")
    static var openAppWhenRun = false

    @Parameter(title: "Action")
    var action: String

    init() {{}}

    func perform() async throws -> some IntentResult {{
        // Perform quick action
        print("Quick action: {{action}}")
        return .result()
    }}
}}
'''

    def _generate_widget_code(
        self,
        project_name: str,
    ) -> str:
        """Generate widget code."""
        return f'''//
//  Widget.swift
//  {project_name}WidgetExtension
//
//  Generated by AI Orchestrator
//

import WidgetKit
import SwiftUI

struct Provider: TimelineProvider {{
    func placeholder(in context: Context) -> SimpleEntry {{
        SimpleEntry(date: Date(), data: "Placeholder")
    }}

    func getSnapshot(in context: Context, completion: @escaping (SimpleEntry) -> ()) {{
        let entry = SimpleEntry(date: Date(), data: "Snapshot")
        completion(entry)
    }}

    func getTimeline(in context: Context, completion: @escaping (Timeline<SimpleEntry>) -> ()) {{
        let entry = SimpleEntry(date: Date(), data: "Timeline")
        let timeline = Timeline(entries: [entry], policy: .atEnd)
        completion(timeline)
    }}
}}

struct SimpleEntry: TimelineEntry {{
    let date: Date
    let data: String
}}

struct {project_name.replace(" ", "")}WidgetEntryView: View {{
    var entry: Provider.Entry

    var body: some View {{
        VStack {{
            Text(entry.date, style: .time)
            Text(entry.data)
        }}
    }}
}}

@main
struct {project_name.replace(" ", "")}Widget: Widget {{
    let kind: String = "{project_name.replace(" ", "")}Widget"

    var body: some WidgetConfiguration {{
        StaticConfiguration(
            kind: kind,
            provider: Provider(),
            content: {{ entry in
                {project_name.replace(" ", "")}WidgetEntryView(entry: entry)
            }}
        )
        .configurationDisplayName("{project_name} Widget")
        .description("View quick information from {project_name}.")
        .supportedFamilies([.systemSmall, .systemMedium])
    }}
}}
'''


# ─────────────────────────────────────────────
# Convenience Function
# ─────────────────────────────────────────────

async def generate_native_feature_template(
    feature: str,
    project_name: str,
    bundle_id: str | None = None,
) -> FeatureTemplate | None:
    """
    Convenience function to generate native feature template.

    Args:
        feature: Feature name
        project_name: Project name
        bundle_id: Bundle identifier

    Returns:
        FeatureTemplate or None
    """
    generator = NativeFeatureTemplateGenerator()
    return await generator.generate(feature, project_name, bundle_id)
