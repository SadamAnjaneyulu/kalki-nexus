import os

code = """
FILES = [
    GeneratedFile('README.md', content_readme()),
    GeneratedFile('.gitignore', content_gitignore()),
    GeneratedFile('.env.example', content_env_example()),
    GeneratedFile('requirements.txt', content_requirements()),
    GeneratedFile('pyproject.toml', content_pyproject()),
    GeneratedFile('config.py', content_config_py()),
    GeneratedFile('core/__init__.py', content_core_init()),
    GeneratedFile('core/exceptions.py', content_core_exceptions()),
    GeneratedFile('core/permissions.py', content_core_permissions()),
    GeneratedFile('core/base_tool.py', content_core_base_tool()),
    GeneratedFile('core/base_memory.py', content_core_base_memory()),
    GeneratedFile('core/base_agent.py', content_core_base_agent()),
    GeneratedFile('core/observability.py', content_core_observability()),
    GeneratedFile('core/resilience.py', content_core_resilience()),
    GeneratedFile('core/registry.py', content_core_registry()),
    GeneratedFile('core/channel_adapter.py', content_core_channel_adapter()),
    GeneratedFile('app.py', content_app_py()),
    GeneratedFile('graph.py', content_graph_py()),
    GeneratedFile('agents/__init__.py', content_agents_init()),
    GeneratedFile('agents/supervisor.py', content_agent_supervisor()),
    GeneratedFile('agents/aggregator.py', content_agent_aggregator()),
    GeneratedFile('agents/fallback_agent.py', content_agent_fallback()),
    GeneratedFile('agents/python_agent.py', content_agent_python()),
    GeneratedFile('agents/docker_agent.py', content_agent_docker()),
    GeneratedFile('agents/github_agent.py', content_agent_github()),
    GeneratedFile('agents/mcp_agent.py', content_agent_mcp()),
    GeneratedFile('agents/automation_agent.py', content_agent_automation()),
    GeneratedFile('agents/research_agent.py', content_agent_research()),
    GeneratedFile('agents/quant_agent.py', content_agent_quant()),
    GeneratedFile('prompts/supervisor.md', content_prompt_supervisor()),
    GeneratedFile('prompts/python.md', content_prompt_python()),
    GeneratedFile('prompts/docker.md', content_prompt_docker()),
    GeneratedFile('prompts/github.md', content_prompt_github()),
    GeneratedFile('prompts/mcp.md', content_prompt_mcp()),
    GeneratedFile('prompts/automation.md', content_prompt_automation()),
    GeneratedFile('prompts/research.md', content_prompt_research()),
    GeneratedFile('prompts/quant.md', content_prompt_quant()),
    GeneratedFile('tools/__init__.py', content_tools_init()),
    GeneratedFile('tools/registry.py', content_tool_registry()),
    GeneratedFile('tools/docker_tools.py', content_tool_docker()),
    GeneratedFile('tools/filesystem_tools.py', content_tool_filesystem()),
    GeneratedFile('tools/github_tools.py', content_tool_github()),
    GeneratedFile('tools/browser_tools.py', content_tool_browser()),
    GeneratedFile('tools/terminal_tools.py', content_tool_terminal()),
    GeneratedFile('tools/discord_tools.py', content_tool_discord()),
    GeneratedFile('tools/web_tools.py', content_tool_web()),
    GeneratedFile('mcp/__init__.py', content_mcp_init()),
    GeneratedFile('mcp/registry.py', content_mcp_registry()),
    GeneratedFile('mcp/client.py', content_mcp_client()),
]

def generate_project(root_dir: str, overwrite: bool) -> None:
    root = Path(root_dir).resolve()
    console.print(f"Bootstrapping [bold cyan]{PROJECT_NAME}[/bold cyan] at: {root}")

    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[green]Generating files...", total=len(FILES))
        for gf in FILES:
            target_path = root / gf.relative_path
            if target_path.exists() and not overwrite:
                progress.console.print(f"[yellow]Skipping[/yellow] {gf.relative_path} (already exists)")
                progress.advance(task)
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(gf.content, encoding="utf-8")
            progress.console.print(f"[green]Wrote[/green] {gf.relative_path}")
            progress.advance(task)
    console.print(f"\\n[bold green]Success![/bold green] Generated {len(FILES)} files.")

def main() -> None:
    print("Inside main()")
    parser = argparse.ArgumentParser(description=f"Bootstrap {PROJECT_NAME}")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--root", default=PROJECT_NAME)
    args = parser.parse_args()
    generate_project(args.root, args.overwrite)

print("Script started")
print("FILES:", len(FILES))
if __name__ == '__main__':
    main()
"""

with open('bootstrap_kalki.py', 'a', encoding='utf-8') as f:
    f.write('\\n' + code)
