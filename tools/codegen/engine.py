"""
Code generation engine for rendering Jinja2 templates.

This module provides a code generation engine that manages multiple Jinja2
template environments, registers templates with their associated data, and
generates rendered files.
"""

from jinja2 import Environment, Template
from pathlib import Path
from typing import Any, Dict, List, NamedTuple


class Renders(NamedTuple):
    """Holds data required to render a single template.

    Attributes:
        template_name: Name of the Jinja2 template file.
        render_path: Destination path for the generated file.
        data: Dictionary of template variables to pass during rendering.
    """

    template_name: str
    render_path: Path
    data: Dict[Any, Any]


class Environments(NamedTuple):
    """Represents a registered Jinja2 environment.

    Attributes:
        id: Unique identifier for the environment.
        env: The Jinja2 Environment instance.
        name: Expected to follow the pattern 'env_{id}'.
    """

    id: int
    env: Environment
    name: str


class _EnvEntry:
    """Internal container for a registered environment and its rendering state."""

    def __init__(self, env_obj: Environments) -> None:
        self.id = env_obj.id
        self.name = env_obj.name
        self.env = env_obj.env
        self.templates: List[Template] = []
        self.renders: List[Renders] = []


class CodeGenEngine:
    """Manages code generation across multiple template environments.

    The engine maintains a registry of environments, allows templates and their
    associated data to be registered, and generates rendered files on demand.

    Example:
        >>> env = Environment(loader=FileSystemLoader('templates'))
        >>> envs = [Environments(id=1, env=env, name='env_1')]
        >>> engine = CodeGenEngine(environments=envs)
        >>> renders = [Renders('template.py', Path('output.py'), {})]
        >>> engine.add_rendering_templates(envs, [renders])
        >>> engine.generate()
    """

    def __init__(self, environments: List[Environments] | None = None) -> None:
        """Initialize the code generation engine.

        Args:
            environments: Optional list of environments to register at startup.
        """
        self._registry: Dict[int, _EnvEntry] = {}

        if environments is not None:
            for env in environments:
                self._register(env)

    def _register(self, env: Environments) -> None:
        """Register a new environment in the engine.

        Args:
            env: The environment to register.

        Raises:
            ValueError: If env is None, the id is duplicate, or name format is invalid.
        """
        if env is None:
            raise ValueError("Environment cannot be None")
        if env.id in self._registry:
            raise ValueError(f"Duplicate environment id: {env.id}")
        if env.name != f"env_{env.id}":
            raise ValueError(
                f"Invalid environment name '{env.name}', expected 'env_{env.id}'"
            )

        self._registry[env.id] = _EnvEntry(env)

    def set_new_env(self, environments: List[Environments]) -> None:
        """Register additional environments.

        Args:
            environments: List of environments to register.
        """
        for env in environments:
            self._register(env)

    def add_rendering_templates(
        self,
        environments: List[Environments],
        renders: List[List[Renders]],
        **template_kwargs,
    ) -> None:
        """Add templates and their rendering data for specified environments.

        Args:
            environments: List of environments to add templates to.
            renders: List of render specifications, one per environment.
            template_kwargs: Additional arguments passed to get_template.

        Raises:
            ValueError: If lengths mismatch or environment is not registered.
        """
        if len(environments) != len(renders):
            raise ValueError(
                f"environments and renders must have the same length, "
                f"got {len(environments)} and {len(renders)}"
            )

        for env, env_renders in zip(environments, renders):
            if env.id not in self._registry:
                raise ValueError(f"Environment id {env.id} not registered")

            entry = self._registry[env.id]
            for trender in env_renders:
                entry.templates.append(
                    env.env.get_template(trender.template_name, **template_kwargs)
                )
                entry.renders.append(trender)

    def _generate_rendered_file(self, path: Path, result: str) -> None:
        """Write rendered content to a file, creating parent directories as needed.

        Args:
            path: Destination file path.
            result: Rendered content to write.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(result)

    def generate(self) -> None:
        """Render all registered templates and write output files.

        Iterates through all registered environments and renders their templates
        using the provided data, writing results to the specified paths.
        """
        for entry in self._registry.values():
            if not entry.templates:
                continue

            for template, render in zip(entry.templates, entry.renders):
                result = template.render(rules=render.data)
                self._generate_rendered_file(render.render_path, result)
