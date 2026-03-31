"""
Manual test for color modification regex - no pytest required
"""

import re


def test_color_changes():
    """Test all color change scenarios"""

    test_cases = [
        {
            "name": "Gold to Purple",
            "css": ":root { --accent: #c9a55c; }",
            "new": "#8b5cf6",
            "expected": "--accent: #8b5cf6",
        },
        {
            "name": "Gold to Blue",
            "css": ":root { --accent: #c9a55c; }",
            "new": "#3b82f6",
            "expected": "--accent: #3b82f6",
        },
        {
            "name": "Gold to Green",
            "css": ":root { --accent: #c9a55c; }",
            "new": "#10b981",
            "expected": "--accent: #10b981",
        },
        {
            "name": "Gold to Black",
            "css": ":root { --accent: #c9a55c; }",
            "new": "#000000",
            "expected": "--accent: #000000",
        },
        {
            "name": "Gold to White",
            "css": ":root { --accent: #c9a55c; }",
            "new": "#ffffff",
            "expected": "--accent: #ffffff",
        },
        {
            "name": "Multiple spaces",
            "css": ":root { --accent:   #c9a55c; }",
            "new": "#8b5cf6",
            "expected": "--accent: #8b5cf6",
        },
        {
            "name": "No space",
            "css": ":root { --accent:#c9a55c; }",
            "new": "#8b5cf6",
            "expected": "--accent:#8b5cf6",
        },
        {
            "name": "3-digit hex",
            "css": ":root { --accent: #fff; }",
            "new": "#000",
            "expected": "--accent: #000",
        },
        {
            "name": "Uppercase hex",
            "css": ":root { --accent: #C9A55C; }",
            "new": "#8B5CF6",
            "expected": "--accent: #8B5CF6",
        },
        {
            "name": "No accent color",
            "css": ":root { --primary: #0a0a0f; }",
            "new": "#8b5cf6",
            "expected": None,  # Should not match
        },
    ]

    accent_pattern = r"--accent:\s*#[0-9a-fA-F]{3,6}"
    passed = 0
    failed = 0

    for tc in test_cases:
        match = re.search(accent_pattern, tc["css"])

        if tc["expected"] is None:
            # Should not match
            if match is None:
                print(f"✅ PASS: {tc['name']} (correctly no match)")
                passed += 1
            else:
                print(f"❌ FAIL: {tc['name']} - expected no match, got {match.group(0)}")
                failed += 1
        else:
            # Should match and replace
            if match is None:
                print(f"❌ FAIL: {tc['name']} - no match found")
                failed += 1
            else:
                result = re.sub(accent_pattern, f'--accent: {tc["new"]}', tc["css"], count=1)
                if tc["expected"] in result:
                    print(f"✅ PASS: {tc['name']}")
                    passed += 1
                else:
                    print(f"❌ FAIL: {tc['name']} - expected '{tc['expected']}', got '{result}'")
                    failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")

    return failed == 0


if __name__ == "__main__":
    success = test_color_changes()
    exit(0 if success else 1)
