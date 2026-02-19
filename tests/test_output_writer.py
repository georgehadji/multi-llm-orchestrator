"""
Tests for output_writer Code Extractor (Improvement 7).
Covers extract_named_files() and write_extracted_files().
"""
from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.output_writer import extract_named_files, write_extracted_files


# ── extract_named_files ───────────────────────────────────────────────────────

class TestExtractNamedFiles:

    def test_single_file_extracted(self):
        text = (
            "Here is the main file:\n\n"
            "**src/main.py**\n"
            "```python\n"
            "print('hello world')  # enough content here\n"
            "```\n"
        )
        result = extract_named_files(text)
        assert "src/main.py" in result
        assert "print('hello world')" in result["src/main.py"]

    def test_multiple_files_extracted(self):
        text = (
            "**src/App.tsx**\n"
            "```typescript\n"
            "export default function App() { return <div/>; }\n"
            "```\n\n"
            "**src/index.css**\n"
            "```css\n"
            "body { margin: 0; padding: 0; }\n"
            "```\n"
        )
        result = extract_named_files(text)
        assert "src/App.tsx" in result
        assert "src/index.css" in result

    def test_backtick_filename_header(self):
        """Handles **`filename.py`** (with backtick wrapping)."""
        text = (
            "**`utils/helpers.py`**\n"
            "```python\n"
            "def helper(): pass  # enough content here\n"
            "```\n"
        )
        result = extract_named_files(text)
        assert "utils/helpers.py" in result

    def test_unknown_extension_skipped(self):
        text = (
            "**READMEFILE**\n"
            "```\n"
            "# Some readme content that is long enough to pass the minimum\n"
            "```\n"
        )
        result = extract_named_files(text)
        assert "READMEFILE" not in result

    def test_dotfile_extracted(self):
        text = (
            "**.gitignore**\n"
            "```\n"
            "node_modules/\n"
            "__pycache__/\n"
            "```\n"
        )
        result = extract_named_files(text)
        assert ".gitignore" in result

    def test_path_traversal_rejected(self):
        text = (
            "**../../etc/passwd**\n"
            "```\n"
            "root:x:0:0:root:/root:/bin/bash\n"
            "```\n"
        )
        result = extract_named_files(text)
        assert len(result) == 0

    def test_normal_path_without_leading_slash(self):
        """Normal LLM output format: **src/main.py** (no leading slash)."""
        text = (
            "**src/main.py**\n"
            "```python\n"
            "x = 1  # content long enough to pass limit\n"
            "```\n"
        )
        result = extract_named_files(text)
        assert "src/main.py" in result

    def test_empty_code_block_skipped(self):
        text = (
            "**src/empty.py**\n"
            "```python\n"
            "\n"
            "```\n"
        )
        result = extract_named_files(text)
        assert "src/empty.py" not in result

    def test_too_short_content_skipped(self):
        text = (
            "**src/tiny.py**\n"
            "```python\n"
            "x = 1\n"
            "```\n"
        )
        result = extract_named_files(text)
        # "x = 1" = 5 chars < 20 _MIN_CONTENT_LEN
        assert "src/tiny.py" not in result

    def test_last_write_wins_for_duplicate_paths(self):
        text = (
            "**src/app.py**\n"
            "```python\n"
            "# first version — this is long enough content block\n"
            "```\n\n"
            "**src/app.py**\n"
            "```python\n"
            "# second version — this is long enough content block\n"
            "```\n"
        )
        result = extract_named_files(text)
        assert "src/app.py" in result
        assert "second version" in result["src/app.py"]

    def test_no_code_fence_skipped(self):
        text = "**src/main.py**\nSome prose without a code fence.\n"
        result = extract_named_files(text)
        assert "src/main.py" not in result

    def test_empty_text_returns_empty_dict(self):
        assert extract_named_files("") == {}

    def test_prose_without_headers_returns_empty_dict(self):
        assert extract_named_files("Just some text without any file headers.") == {}

    def test_yaml_file_extracted(self):
        text = (
            "**docker-compose.yml**\n"
            "```yaml\n"
            "version: '3'\nservices:\n  app:\n    image: myapp\n"
            "```\n"
        )
        result = extract_named_files(text)
        assert "docker-compose.yml" in result

    def test_nested_path_preserved(self):
        text = (
            "**src/components/Button/index.tsx**\n"
            "```tsx\n"
            "export const Button = () => <button>Click</button>;\n"
            "```\n"
        )
        result = extract_named_files(text)
        assert "src/components/Button/index.tsx" in result


# ── write_extracted_files ─────────────────────────────────────────────────────

class TestWriteExtractedFiles:

    def test_writes_files_to_output_dir(self, tmp_path):
        files = {
            "src/main.py": "print('hello')\n",
            "README.md": "# My App\n",
        }
        written = write_extracted_files(files, tmp_path)
        assert set(written) == {"src/main.py", "README.md"}
        assert (tmp_path / "src" / "main.py").read_text() == "print('hello')\n"
        assert (tmp_path / "README.md").read_text() == "# My App\n"

    def test_creates_parent_directories(self, tmp_path):
        files = {"a/b/c/deep.ts": "export const x = 1;\n"}
        write_extracted_files(files, tmp_path)
        assert (tmp_path / "a" / "b" / "c" / "deep.ts").exists()

    def test_returns_list_of_written_paths(self, tmp_path):
        files = {"x.py": "x = 1  # enough content here\n"}
        written = write_extracted_files(files, tmp_path)
        assert written == ["x.py"]

    def test_empty_dict_returns_empty_list(self, tmp_path):
        assert write_extracted_files({}, tmp_path) == []

    def test_content_preserved_exactly(self, tmp_path):
        content = (
            "import React from 'react';\n\n"
            "export default function App() {\n"
            "  return <div>Hello</div>;\n"
            "}\n"
        )
        write_extracted_files({"App.tsx": content}, tmp_path)
        assert (tmp_path / "App.tsx").read_text(encoding="utf-8") == content
