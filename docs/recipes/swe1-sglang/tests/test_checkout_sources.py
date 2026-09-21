"""Offline transport fixtures and rejection tests for the source-only bootstrap."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


RECIPE = Path(__file__).resolve().parents[1]
GYM_PATH = "3rdparty/Gym-workspace/Gym"
NEMO_URL = "https://github.com/Kh4L/NemoRL.git"
GYM_URL = "https://github.com/Kh4L/NemoGym.git"


def run_git(directory: Path, *arguments: str, env: dict[str, str]) -> str:
    """Run Git against an isolated test repository and return its output."""
    result = subprocess.run(
        ["git", "-C", str(directory), *arguments],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


class CheckoutSourcesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture_directory = tempfile.TemporaryDirectory(
            prefix="swe1-public-fixtures-"
        )
        cls.addClassCleanup(cls.fixture_directory.cleanup)
        cls.fixture_root = Path(cls.fixture_directory.name)
        cls.env = os.environ.copy()
        for key in (
            "GIT_DIR",
            "GIT_WORK_TREE",
            "GIT_INDEX_FILE",
            "GIT_OBJECT_DIRECTORY",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES",
            "GIT_CONFIG_COUNT",
            "GIT_CONFIG_PARAMETERS",
            "GIT_CONFIG",
        ):
            cls.env.pop(key, None)
        for key in list(cls.env):
            if key.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")):
                cls.env.pop(key)
        cls.env.update(
            {
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_AUTHOR_NAME": "Serge Panev",
                "GIT_AUTHOR_EMAIL": "spanev@nvidia.com",
                "GIT_COMMITTER_NAME": "Serge Panev",
                "GIT_COMMITTER_EMAIL": "spanev@nvidia.com",
                "GIT_TERMINAL_PROMPT": "0",
            }
        )
        cls.gym = cls.fixture_root / "gym"
        cls.nemo = cls.fixture_root / "nemo"
        for repository in (cls.gym, cls.nemo):
            repository.mkdir()
            run_git(repository, "init", "--initial-branch=main", env=cls.env)
        (cls.gym / "README.md").write_text("Offline Gym source fixture.\n")
        run_git(cls.gym, "add", "README.md", env=cls.env)
        run_git(
            cls.gym,
            "commit",
            "-s",
            "-m",
            "test: create Gym transport fixture",
            env=cls.env,
        )
        cls.gym_commit = run_git(cls.gym, "rev-parse", "HEAD", env=cls.env)
        (cls.nemo / ".gitmodules").write_text(
            f'[submodule "{GYM_PATH}"]\n\tpath = {GYM_PATH}\n'
            "\turl = https://github.com/NVIDIA-NeMo/Gym.git\n"
        )
        (cls.nemo / "README.md").write_text("Offline NeMo source fixture.\n")
        run_git(cls.nemo, "add", ".gitmodules", "README.md", env=cls.env)
        run_git(
            cls.nemo,
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{cls.gym_commit},{GYM_PATH}",
            env=cls.env,
        )
        run_git(
            cls.nemo,
            "commit",
            "-s",
            "-m",
            "test: create NeMo transport fixture",
            env=cls.env,
        )
        cls.runtime = run_git(cls.nemo, "rev-parse", "HEAD", env=cls.env)
        guide = cls.nemo / "docs/guides/swe-rl-qwen3.md"
        guide.parent.mkdir(parents=True)
        guide.write_text("Offline documentation fixture.\n")
        run_git(cls.nemo, "add", "docs/guides/swe-rl-qwen3.md", env=cls.env)
        run_git(cls.nemo, "commit", "-s", "-m", "docs: add fixture guide", env=cls.env)
        cls.docs = run_git(cls.nemo, "rev-parse", "HEAD", env=cls.env)
        # Only tests redirect the public transports to local fixture repositories.
        # No override option or test-only URL exists in the production bootstrap.
        cls.env.update(
            {
                "GIT_CONFIG_COUNT": "3",
                "GIT_CONFIG_KEY_0": f"url.{cls.nemo.as_uri()}.insteadOf",
                "GIT_CONFIG_VALUE_0": NEMO_URL,
                "GIT_CONFIG_KEY_1": f"url.{cls.gym.as_uri()}.insteadOf",
                "GIT_CONFIG_VALUE_1": GYM_URL,
                "GIT_CONFIG_KEY_2": "protocol.file.allow",
                "GIT_CONFIG_VALUE_2": "always",
            }
        )

    def setUp(self) -> None:
        self.scratch = tempfile.TemporaryDirectory(prefix="swe1-public-checkout-")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.recipe = self.root / "recipe"
        (self.recipe / "scripts").mkdir(parents=True)
        shutil.copy2(
            RECIPE / "scripts/checkout_sources.sh",
            self.recipe / "scripts/checkout_sources.sh",
        )
        self.lock = self.recipe / "SOURCES.lock"
        self.lock_text = (
            "# Offline test fixture pins.\nFORMAT_VERSION=1\n"
            f"NEMO_URL={NEMO_URL}\nNEMO_RUNTIME={self.runtime}\nNEMO_DOCS={self.docs}\n"
            f"GYM_URL={GYM_URL}\nGYM_COMMIT={self.gym_commit}\n"
        )
        self.lock.write_text(self.lock_text)
        self.destination = self.root / "source"

    def invoke(
        self, *arguments: str, env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(self.recipe / "scripts/checkout_sources.sh"), *arguments],
            env=self.env if env is None else env,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def assert_rejected(self, result: subprocess.CompletedProcess[str]) -> None:
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.destination.exists())

    def check_success(
        self, *, profile: str, expected: str, env: dict[str, str] | None = None
    ) -> None:
        result = self.invoke(str(self.destination), profile, env=env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            run_git(self.destination, "rev-parse", "HEAD", env=self.env), expected
        )
        self.assertEqual(
            run_git(self.destination / GYM_PATH, "rev-parse", "HEAD", env=self.env),
            self.gym_commit,
        )
        self.assertEqual(
            run_git(self.destination, "status", "--porcelain", env=self.env), ""
        )
        self.assertEqual(
            run_git(
                self.destination, "config", f"submodule.{GYM_PATH}.url", env=self.env
            ),
            GYM_URL,
        )
        self.assertIn("No environments installed or training launched.", result.stdout)

    def test_docs_profile_restores_exact_source_and_gym(self) -> None:
        self.check_success(profile="docs", expected=self.docs)

    def test_runtime_profile_restores_exact_source_and_gym(self) -> None:
        self.check_success(profile="runtime", expected=self.runtime)

    def test_default_profile_is_docs(self) -> None:
        result = self.invoke(str(self.destination))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            run_git(self.destination, "rev-parse", "HEAD", env=self.env), self.docs
        )

    def test_ambient_git_paths_do_not_redirect_checkout(self) -> None:
        env = self.env | {
            "GIT_DIR": str(self.root / "wrong-git"),
            "GIT_WORK_TREE": str(self.root / "wrong-tree"),
            "GIT_INDEX_FILE": str(self.root / "wrong-index"),
            "GIT_OBJECT_DIRECTORY": str(self.root / "wrong-objects"),
            "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(self.root / "wrong-alternates"),
        }
        self.check_success(profile="docs", expected=self.docs, env=env)
        self.assertFalse((self.root / "wrong-index").exists())

    def test_argument_and_destination_rejections(self) -> None:
        for arguments in (
            (),
            (str(self.destination), "docs", "extra"),
            ("relative",),
            ("/",),
            (str(self.destination) + "/",),
            (str(self.root / "absent/source"),),
            (str(self.destination), "invalid"),
        ):
            with self.subTest(arguments=arguments):
                self.assert_rejected(self.invoke(*arguments))

    def test_existing_directory_is_preserved(self) -> None:
        self.destination.mkdir()
        marker = self.destination / "keep"
        marker.write_text("unchanged\n")
        result = self.invoke(str(self.destination))
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(marker.read_text(), "unchanged\n")
        self.assertFalse((self.destination / ".git").exists())

    def test_dangling_destination_symlink_is_preserved(self) -> None:
        self.destination.symlink_to(self.root / "absent")
        result = self.invoke(str(self.destination))
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(self.destination.is_symlink())
        self.assertFalse((self.root / "absent").exists())

    def test_malformed_lock_rejections(self) -> None:
        mutations = {
            "duplicate": self.lock_text + "FORMAT_VERSION=1\n",
            "unknown": self.lock_text + "SURPRISE=1\n",
            "malformed": self.lock_text + "no equals\n",
            "missing": self.lock_text.replace(f"GYM_COMMIT={self.gym_commit}\n", ""),
            "empty": self.lock_text.replace(
                f"GYM_COMMIT={self.gym_commit}", "GYM_COMMIT="
            ),
            "format": self.lock_text.replace("FORMAT_VERSION=1", "FORMAT_VERSION=2"),
            "floating": self.lock_text.replace(self.docs, "main"),
            "bad_sha": self.lock_text.replace(self.gym_commit, "x" * 40),
            "nemo_url": self.lock_text.replace(
                NEMO_URL, "https://example.invalid/RL.git"
            ),
            "gym_url": self.lock_text.replace(
                GYM_URL, "https://example.invalid/Gym.git"
            ),
        }
        for name, content in mutations.items():
            with self.subTest(name=name):
                self.lock.write_text(content)
                self.assert_rejected(self.invoke(str(self.destination)))

    def test_lock_shell_expression_is_not_executed(self) -> None:
        marker = self.root / "must-not-exist"
        self.lock.write_text(
            self.lock_text.replace(self.gym_commit, f"$(touch {marker})")
        )
        self.assert_rejected(self.invoke(str(self.destination)))
        self.assertFalse(marker.exists())

    def test_lock_symlink_is_rejected(self) -> None:
        actual = self.root / "actual.lock"
        self.lock.rename(actual)
        self.lock.symlink_to(actual)
        self.assert_rejected(self.invoke(str(self.destination)))

    def test_wrong_gym_gitlink_fails_and_retains_partial_source(self) -> None:
        self.lock.write_text(self.lock_text.replace(self.gym_commit, self.runtime))
        result = self.invoke(str(self.destination))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Pinned Gym gitlink differs", result.stderr)
        self.assertIn("partial destination retained", result.stderr)
        self.assertTrue((self.destination / ".git").is_dir())
        self.assertFalse((self.destination / GYM_PATH / ".git").exists())

    def test_unavailable_source_fails_and_retains_partial_source(self) -> None:
        self.lock.write_text(self.lock_text.replace(self.docs, "0" * 40))
        result = self.invoke(str(self.destination))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("partial destination retained", result.stderr)
        self.assertTrue((self.destination / ".git").is_dir())
        self.assertNotIn("Source checkout verified:", result.stdout)


if __name__ == "__main__":
    unittest.main()
