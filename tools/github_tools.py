"""
Kalki Nexus - GitHub Tools

BaseTool wrappers around the GitHub REST API via PyGithub, registered under
the "github" category. Requires GITHUB_TOKEN to be set in the environment.
"""
from __future__ import annotations

from github import Github

from config import get_settings
from core.base_tool import BaseTool
from core.permissions import Permission
from tools.registry import ToolRegistry


def _client() -> Github:
    settings = get_settings()
    if not settings.github_token:
        raise RuntimeError("GITHUB_TOKEN is not set. Copy .env.example to .env and fill it in.")
    return Github(settings.github_token)


@ToolRegistry.register()
class ListOpenPullRequestsTool(BaseTool):
    name = "list_open_pull_requests"
    description = "List the titles of open pull requests for owner/repo."
    category = "github"
    required_permissions = [Permission.GITHUB, Permission.NETWORK]

    async def run(self, repo_full_name: str) -> list:
        repo = _client().get_repo(repo_full_name)
        return [pr.title for pr in repo.get_pulls(state="open")]


@ToolRegistry.register()
class CreateIssueTool(BaseTool):
    name = "create_issue"
    description = "Create an issue in owner/repo and return its URL."
    category = "github"
    required_permissions = [Permission.GITHUB, Permission.NETWORK]

    async def run(self, repo_full_name: str, title: str, body: str = "") -> str:
        repo = _client().get_repo(repo_full_name)
        issue = repo.create_issue(title=title, body=body)
        return issue.html_url


@ToolRegistry.register()
class GetFileContentsTool(BaseTool):
    name = "get_file_contents"
    description = "Fetch the decoded text contents of a file at a path in owner/repo."
    category = "github"
    required_permissions = [Permission.GITHUB, Permission.NETWORK]

    async def run(self, repo_full_name: str, path: str, ref: str = "main") -> str:
        repo = _client().get_repo(repo_full_name)
        content_file = repo.get_contents(path, ref=ref)
        return content_file.decoded_content.decode("utf-8")


# TODO: add tools for merging PRs, requesting reviews, and posting review
# comments - route them through GithubAgent's destructive-action detection
# (agents/github_agent.py) so they require human approval.
