"""
BrowserTesting — Browser-based testing
===================================
Module for performing browser-based testing using Playwright or similar tools.

Pattern: Facade
Async: Yes — for I/O-bound browser operations
Layer: L3 Agents

Usage:
    from orchestrator.browser_testing import BrowserTester
    tester = BrowserTester(browser_type="chromium")
    result = await tester.test_page("https://example.com", [
        {"action": "click", "selector": "#button"},
        {"action": "fill", "selector": "#input", "value": "test"}
    ])
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("orchestrator.browser_testing")


@dataclass
class TestStep:
    """Represents a single step in a browser test."""

    action: str  # click, fill, navigate, assert, etc.
    selector: str | None = None
    value: str | None = None
    expected: str | None = None
    timeout: int = 30000  # milliseconds


@dataclass
class TestResult:
    """Represents the result of a browser test."""

    success: bool
    steps_executed: int
    steps_passed: int
    errors: list[str]
    screenshots: list[str]  # Paths to screenshots taken during test
    execution_time: float
    details: dict[str, Any]


class BrowserTester:
    """Performs browser-based testing using Playwright or similar tools."""

    def __init__(self, browser_type: str = "chromium", headless: bool = True,
                 screenshot_on_failure: bool = True):
        """
        Initialize the browser tester.

        Args:
            browser_type: Type of browser to use ("chromium", "firefox", "webkit")
            headless: Whether to run browser in headless mode
            screenshot_on_failure: Whether to take screenshots on failures
        """
        self.browser_type = browser_type
        self.headless = headless
        self.screenshot_on_failure = screenshot_on_failure
        self.browser = None
        self.context = None
        self.page = None
        self.screenshots_taken = []

    async def initialize(self):
        """Initialize the browser and create a context."""
        try:
            from playwright.async_api import async_playwright

            self.playwright = await async_playwright().start()

            # Launch browser
            if self.browser_type == "chromium":
                self.browser = await self.playwright.chromium.launch(headless=self.headless)
            elif self.browser_type == "firefox":
                self.browser = await self.playwright.firefox.launch(headless=self.headless)
            elif self.browser_type == "webkit":
                self.browser = await self.playwright.webkit.launch(headless=self.headless)
            else:
                raise ValueError(f"Unsupported browser type: {self.browser_type}")

            # Create context
            self.context = await self.browser.new_context()

            # Create page
            self.page = await self.context.new_page()

            logger.info(f"Browser tester initialized with {self.browser_type}")
        except ImportError:
            logger.error("Playwright not installed. Install with: pip install playwright")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize browser tester: {e}")
            raise

    async def close(self):
        """Close the browser and clean up resources."""
        if self.page:
            await self.page.close()
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if hasattr(self, 'playwright'):
            await self.playwright.stop()

        logger.info("Browser tester closed")

    async def test_page(self, url: str, test_steps: list[TestStep],
                       wait_until: str = "networkidle") -> TestResult:
        """
        Test a page by executing a series of steps.

        Args:
            url: URL to navigate to
            test_steps: List of steps to execute
            wait_until: When to consider navigation succeeded ("load", "domcontentloaded", "networkidle")

        Returns:
            TestResult: Result of the test execution
        """
        import time

        if not self.page:
            await self.initialize()

        start_time = time.time()
        errors = []
        steps_passed = 0
        self.screenshots_taken = []

        try:
            # Navigate to the page
            await self.page.goto(url, wait_until=wait_until)

            # Execute each test step
            for i, step in enumerate(test_steps):
                try:
                    await self._execute_step(step)
                    steps_passed += 1
                    logger.debug(f"Step {i+1} passed: {step.action}")
                except Exception as e:
                    error_msg = f"Step {i+1} failed ({step.action}): {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)

                    # Take screenshot on failure if enabled
                    if self.screenshot_on_failure:
                        screenshot_path = await self._take_screenshot(f"failure_step_{i+1}")
                        self.screenshots_taken.append(screenshot_path)

        except Exception as e:
            errors.append(f"Navigation error: {str(e)}")
            logger.error(f"Navigation error: {e}")

            # Take screenshot on failure if enabled
            if self.screenshot_on_failure:
                screenshot_path = await self._take_screenshot("navigation_failure")
                self.screenshots_taken.append(screenshot_path)

        execution_time = time.time() - start_time

        return TestResult(
            success=len(errors) == 0,
            steps_executed=len(test_steps),
            steps_passed=steps_passed,
            errors=errors,
            screenshots=self.screenshots_taken,
            execution_time=execution_time,
            details={
                "url": url,
                "browser_type": self.browser_type,
                "headless": self.headless
            }
        )

    async def _execute_step(self, step: TestStep):
        """Execute a single test step."""
        if step.action == "click":
            if not step.selector:
                raise ValueError("Click action requires a selector")
            await self.page.click(step.selector, timeout=step.timeout)

        elif step.action == "fill":
            if not step.selector or not step.value:
                raise ValueError("Fill action requires both selector and value")
            await self.page.fill(step.selector, step.value, timeout=step.timeout)

        elif step.action == "navigate":
            if not step.value:
                raise ValueError("Navigate action requires a URL value")
            await self.page.goto(step.value, timeout=step.timeout)

        elif step.action == "assert_text":
            if not step.selector or not step.expected:
                raise ValueError("Assert text action requires both selector and expected value")
            element = await self.page.wait_for_selector(step.selector, timeout=step.timeout)
            actual_text = await element.text_content()
            if step.expected not in actual_text:
                raise AssertionError(f"Expected '{step.expected}' in text '{actual_text}'")

        elif step.action == "assert_visible":
            if not step.selector:
                raise ValueError("Assert visible action requires a selector")
            await self.page.wait_for_selector(step.selector, state="visible", timeout=step.timeout)

        elif step.action == "assert_not_visible":
            if not step.selector:
                raise ValueError("Assert not visible action requires a selector")
            await self.page.wait_for_selector(step.selector, state="hidden", timeout=step.timeout)

        elif step.action == "screenshot":
            filename = step.value or f"screenshot_{int(time.time())}"
            screenshot_path = await self._take_screenshot(filename)
            self.screenshots_taken.append(screenshot_path)

        elif step.action == "hover":
            if not step.selector:
                raise ValueError("Hover action requires a selector")
            await self.page.hover(step.selector, timeout=step.timeout)

        elif step.action == "press":
            if not step.value:
                raise ValueError("Press action requires a key value")
            await self.page.keyboard.press(step.value)

        elif step.action == "scroll":
            # Scroll by pixels or to an element
            if step.selector:
                await self.page.locator(step.selector).scroll_into_view_if_needed(timeout=step.timeout)
            else:
                # Scroll by pixels (value should be a JSON string like '{"x": 0, "y": 500}')
                scroll_params = json.loads(step.value) if step.value else {"x": 0, "y": 500}
                await self.page.mouse.wheel(scroll_params.get("x", 0), scroll_params.get("y", 500))

        else:
            raise ValueError(f"Unknown action: {step.action}")

    async def _take_screenshot(self, name: str) -> str:
        """Take a screenshot and return the path."""
        import os
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.png"
        filepath = os.path.join("screenshots", filename)

        # Create screenshots directory if it doesn't exist
        os.makedirs("screenshots", exist_ok=True)

        await self.page.screenshot(path=filepath)
        logger.info(f"Screenshot taken: {filepath}")

        return filepath

    async def run_test_suite(self, test_suite: list[dict[str, Any]]) -> list[TestResult]:
        """
        Run a suite of browser tests.

        Args:
            test_suite: List of test definitions with 'url' and 'steps' keys

        Returns:
            List of TestResult objects
        """
        results = []

        for test_def in test_suite:
            url = test_def.get("url")
            steps_data = test_def.get("steps", [])

            # Convert step data to TestStep objects
            steps = []
            for step_data in steps_data:
                step = TestStep(
                    action=step_data["action"],
                    selector=step_data.get("selector"),
                    value=step_data.get("value"),
                    expected=step_data.get("expected"),
                    timeout=step_data.get("timeout", 30000)
                )
                steps.append(step)

            result = await self.test_page(url, steps)
            results.append(result)

        return results

    async def check_page_performance(self, url: str, metrics: list[str] = None) -> dict[str, Any]:
        """
        Check page performance metrics.

        Args:
            url: URL to navigate to
            metrics: List of metrics to collect (e.g., ["FCP", "LCP", "CLS"])

        Returns:
            Dict with performance metrics
        """
        if not self.page:
            await self.initialize()

        if metrics is None:
            metrics = ["FCP", "LCP", "CLS", "FID", "TTFB"]

        # Enable performance metrics
        await self.page.goto(url)

        # Execute JavaScript to get performance metrics
        perf_metrics = {}

        if "FCP" in metrics:
            # First Contentful Paint
            fcp_js = """
            new Promise(resolve => {
                const observer = new PerformanceObserver(list => {
                    for (const entry of list.getEntries()) {
                        if (entry.name === 'first-contentful-paint') {
                            observer.disconnect();
                            resolve(entry.startTime);
                        }
                    }
                });
                observer.observe({entryTypes: ['paint']});

                // Fallback if FCP is not observed
                setTimeout(() => resolve(-1), 30000);
            });
            """
            try:
                fcp = await self.page.evaluate(fcp_js)
                perf_metrics["FCP"] = fcp
            except:
                perf_metrics["FCP"] = -1

        if "LCP" in metrics:
            # Largest Contentful Paint
            lcp_js = """
            new Promise(resolve => {
                const observer = new PerformanceObserver(list => {
                    const entries = list.getEntries();
                    const lastEntry = entries[entries.length - 1];
                    resolve(lastEntry.renderTime || lastEntry.loadTime);
                });
                observer.observe({entryTypes: ['largest-contentful-paint']});

                // Report the last LCP entry after a delay
                setTimeout(() => {
                    const entries = performance.getEntriesByType('largest-contentful-paint');
                    if (entries.length > 0) {
                        const lastEntry = entries[entries.length - 1];
                        resolve(lastEntry.renderTime || lastEntry.loadTime);
                    } else {
                        resolve(-1);
                    }
                }, 10000);
            });
            """
            try:
                lcp = await self.page.evaluate(lcp_js)
                perf_metrics["LCP"] = lcp
            except:
                perf_metrics["LCP"] = -1

        if "CLS" in metrics:
            # Cumulative Layout Shift
            cls_js = """
            new Promise(resolve => {
                let clsValue = 0;
                const observer = new PerformanceObserver(list => {
                    for (const entry of list.getEntries()) {
                        if (entry.entryType === 'layout-shift' && !entry.hadRecentInput) {
                            clsValue += entry.value;
                        }
                    }
                });
                observer.observe({entryTypes: ['layout-shift']});

                // Report CLS after a delay
                setTimeout(() => {
                    resolve(clsValue);
                }, 5000);
            });
            """
            try:
                cls = await self.page.evaluate(cls_js)
                perf_metrics["CLS"] = cls
            except:
                perf_metrics["CLS"] = -1

        # Time to First Byte (TTFB)
        if "TTFB" in metrics:
            timing = await self.page.evaluate("JSON.stringify(performance.timing)")
            timing_obj = json.loads(timing)
            ttfb = timing_obj["responseStart"] - timing_obj["requestStart"]
            perf_metrics["TTFB"] = ttfb

        return perf_metrics

    async def check_accessibility(self, url: str) -> dict[str, Any]:
        """
        Check accessibility of a page using axe-core.

        Args:
            url: URL to navigate to

        Returns:
            Dict with accessibility violations
        """
        if not self.page:
            await self.initialize()

        # Navigate to the page
        await self.page.goto(url)

        # Inject axe-core script
        axe_script = """
        (function(){
            // This is a simplified version - in practice, you'd load the full axe-core library
            const violations = [];

            // Check for common accessibility issues
            const images = Array.from(document.querySelectorAll('img'));
            for (const img of images) {
                if (!img.alt) {
                    violations.push({
                        id: 'image-alt-missing',
                        impact: 'critical',
                        description: 'Image missing alt attribute',
                        nodes: [{ target: [img.tagName + (img.id ? '#' + img.id : '')] }]
                    });
                }
            }

            // Check for sufficient color contrast
            const elements = Array.from(document.querySelectorAll('*'));
            for (const el of elements) {
                // Simplified contrast check
                const style = window.getComputedStyle(el);
                const bgColor = style.backgroundColor;
                const textColor = style.color;

                // This is a very simplified check - real implementation would be more complex
                if (el.textContent && el.textContent.trim() !== '' &&
                    (bgColor === textColor || bgColor === 'rgba(0, 0, 0, 0)')) {
                    violations.push({
                        id: 'color-contrast-insufficient',
                        impact: 'serious',
                        description: 'Possible insufficient color contrast',
                        nodes: [{ target: [el.tagName + (el.id ? '#' + el.id : '')] }]
                    });
                }
            }

            return { violations };
        })();
        """

        try:
            result = await self.page.evaluate(axe_script)
            return result
        except Exception as e:
            logger.error(f"Accessibility check failed: {e}")
            return {"violations": [], "error": str(e)}

    def get_browser_info(self) -> dict[str, str]:
        """
        Get information about the current browser.

        Returns:
            Dict with browser information
        """
        if not self.browser:
            return {"status": "not_initialized"}

        return {
            "browser_type": self.browser_type,
            "headless": self.headless,
            "is_connected": self.browser.is_connected() if self.browser else False
        }


# Global browser tester for the orchestrator
_global_browser_tester: BrowserTester | None = None


async def get_global_browser_tester(browser_type: str = "chromium",
                                  headless: bool = True) -> BrowserTester:
    """
    Get the global browser tester instance, creating it if it doesn't exist.

    Args:
        browser_type: Type of browser to use
        headless: Whether to run in headless mode

    Returns:
        BrowserTester instance
    """
    global _global_browser_tester
    if _global_browser_tester is None:
        _global_browser_tester = BrowserTester(browser_type=browser_type, headless=headless)
        await _global_browser_tester.initialize()
    return _global_browser_tester


async def close_global_browser_tester():
    """Close the global browser tester instance."""
    global _global_browser_tester
    if _global_browser_tester:
        await _global_browser_tester.close()
        _global_browser_tester = None
