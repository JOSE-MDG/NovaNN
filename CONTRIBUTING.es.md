# Guía de Contribución

Gracias por tu interés en contribuir a **NovaNN**.

NovaNN está pasando a v5. v4 era un framework de Python y NumPy creado para
entender cómo funcionan internamente los frameworks de deep learning. Ese
objetivo educativo sigue presente, pero ya no es el único. v5 mantiene la API de
Python, reemplaza NumPy con un núcleo nativo bajo `ncore/`, escrito en C23,
C++23 y Rust, y amplía el alcance desde entrenar modelos hasta cubrir
toda la vida de un modelo: pretraining, fine-tuning, quantización, adapters,
inferencia y despliegue en CPU, CUDA e HIP. El paquete Python de `nova/` sigue
siendo la versión estable v4.0.4 y se publica todavía. Los cambios en esa parte
son mantenimiento legacy, no la dirección del proyecto.

## Tabla de Contenidos

- [Código de Conducta](#código-de-conducta)
- [¿Cómo Puedo Contribuir?](#cómo-puedo-contribuir)
- [Configuración del Entorno de Desarrollo](#configuración-del-entorno-de-desarrollo)
- [Trabajar con Agentes](#trabajar-con-agentes)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Estándares de Código](#estándares-de-código)
- [Commits](#commits)
- [Testing](#testing)
- [CI](#ci)
- [Proceso de Pull Request](#proceso-de-pull-request)
- [Reportar Bugs](#reportar-bugs)
- [Proponer Features](#proponer-features)
- [Preguntas Frecuentes](#preguntas-frecuentes)

## Código de Conducta

Este proyecto busca ser un espacio acogedor, colaborativo y profesional.
Se espera:

- **Respeto mutuo**: trata a todos con cortesía y profesionalismo
- **Crítica constructiva**: enfócate en el código, no en las personas
- **Colaboración**: ayuda a otros a aprender y crecer
- **Paciencia**: da feedback claro y accionable

## ¿Cómo Puedo Contribuir?

### Áreas Prioritarias

El trabajo actual se centra en el núcleo nativo y el tooling que lo rodea:

- Núcleo nativo
- Backends de CPU y GPU
- Gestión de memoria
- Generación de código y tooling de build
- Tests y documentación

Áreas hacia las que se dirige el ecosistema, no todas implementadas aún:

- Arquitecturas transformer y manejo de datasets
- Flujos de quantización como NF4 y FP8
- Fine-tuning eficiente con adapters como LoRA y QLoRA
- Trainers de SFT y DPO
- Inferencia, despliegue y un model hub

El orden y el estado de estas áreas se siguen en `v5-notes.md`.

### Proceso General

1. Haz un fork del repositorio en GitHub.
2. Clona tu fork con los submodules.
3. Crea una rama para tu cambio: `git checkout -b feat/nueva-funcionalidad`.
   Usa `feat/`, `fix/`, `infra/` o `ci/` para que el propósito sea claro.
4. Haz el cambio siguiendo los estándares de abajo.
5. Haz commit con un mensaje de Conventional Commits.
6. Haz push a tu fork: `git push origin feat/nueva-funcionalidad`.
7. Abre un pull request contra `dev`.
8. Con CI en verde y el cambio aprobado, abre un pull request de `dev` a
   `main` y después el pull request correspondiente al repositorio upstream.

Abrir un pull request directamente contra `main` lo rechaza el check
`branch-origin`. Todo cambio pasa por `dev`.

## Configuración del Entorno de Desarrollo

### Requisitos Previos

- CMake >= 3.27 y Ninja
- Linux: GCC >= 15. La imagen `ubuntu-26.04` fijada en CI ya lo incluye.
- Windows: clang-cl y lld-link >= 20.1. MSVC `cl.exe` no es compilador del
  proyecto, salvo como host compiler de nvcc en Windows.
- Toolchain de Rust para `ncore/memory`
- Python >= 3.12 con `uv`. `.python-version` fija 3.14.4.
- Git con soporte de submodules. `thirdParty/vcpkg` está fijado por commit.
- Opcional: CUDA o HIP, que son mutuamente excluyentes en configuración
  (`-DUSE_CUDA=ON` o `-DUSE_HIP=ON`). HIP solo está disponible en Linux.

### Instalación

```bash
git clone --recurse-submodules git@github.com:JOSE-MDG/NovaNN.git
cd NovaNN
```

Un clone normal también funciona: `scripts/lib/vcpkg-env.sh`, o su equivalente
en PowerShell, inicializa y compila el submodule de vcpkg fijado cuando hace
falta.

### Compilar y Testear

```bash
scripts/build-presets.sh cpu-test-release-linux
scripts/compile-presets.sh -j "$(nproc)" cpu-test-release-linux
scripts/run-tests.sh cpu-test-release-linux
```

En Windows usa los scripts de PowerShell con un preset `*-windows`:

```powershell
.\scripts\build-presets.ps1 cpu-test-release-windows
.\scripts\compile-presets.ps1 -j 4 cpu-test-release-windows
.\scripts\run-tests.ps1 --force-stop-on-failure cpu-test-release-windows
```

`cmake --list-presets` muestra todos los presets disponibles.
`cmake --workflow --preset <preset>` encadena configuración, compilación y
tests en un solo comando. Los nombres siguen
`<backend>-<config>[-<sanitizer>][-test][-os]`, por ejemplo
`cpu-asan-test-debug-linux` o `hip-ubsan-test-release-linux`.

## Trabajar con Agentes

v5 se desarrolla con ayuda de agentes de IA. Humanos y agentes siguen las
mismas reglas, documentadas en `AGENTS.md`. El repo incluye skills reutilizables
bajo `.agents/skills/`, con definiciones de agentes replicadas en
`.github/agents/`, `.opencode/agents/` y `.claude/agents/`. Los agentes del
proyecto incluyen `commit-creator`, `implementation-review`, `test-creator`,
`test-integrity` y `build-integrity`.

El trabajo parte de una spec: una spec de feature bajo `specs/` se escribe antes
de implementar. Los archivos generados se regeneran desde templates en lugar de
editarse a mano. Los agentes informan de sus hallazgos con referencias
`file:line`, mantienen las respuestas concisas y las escriben en inglés.

Los cambios de un agente pasan por el mismo gate que los de una persona:
borrador, revisión, tests y pull request a `dev`. Un agente nunca hace push,
rebase, resolución de conflictos ni aplica un commit sin aprobación explícita.

## Estructura del Proyecto

```
NovaNN/
├── ncore/
│   ├── include/           # Headers públicos C y C++ header-only
│   ├── src/               # Implementación C y C++
│   ├── native/            # Backends CPU, CUDA, HIP y kernels
│   ├── memory/            # Allocator de Rust (ncore_memory)
│   └── tests/             # Suites GTest registradas en CTest
├── tools/codegen/         # Templates, reglas y punto de entrada
├── scripts/               # Scripts de build, test y CI
├── .github/workflows/     # Pipelines de GitHub Actions
├── nova/                  # Paquete Python legacy v4.0.4
├── tests/                 # Suite pytest legacy
├── examples/              # Scripts de ejemplo
└── benchmarks/            # Benchmarks de rendimiento
```

### ¿Dónde Va Cada Cosa?

- **Nuevo tipo u operación del core**: `ncore/include/` y `ncore/src/`
- **Nuevo kernel de backend**: `ncore/native/` con su caso GTest en `ncore/tests/`
- **Nueva primitiva de memoria**: `ncore/memory/`
- **Nueva tabla generada**: un template y una regla bajo `tools/codegen/`
- **Nuevo caso GTest**: `ncore/tests/`, espejando el módulo que cubre
- **Cambio Python legacy**: `nova/` y `tests/`

## Estándares de Código

### C y C++

- Escribe C23 y C++23. Los comentarios Doxygen usan resúmenes de una línea;
  añade detalles solo cuando la razón no se deduce de la firma.
- Usa initializers designados para structs y tablas, un miembro por línea.
- Los headers compartidos con C++ hacen casting mediante `STATIC_CAST` de
  `ncore/include/ncore/headeronly/macros.h`.
- Sufijos de headers: `.h` para C, `.hh` para C++ header-only, `.hpp` para
  utilidades del árbol de tests.
- Los comentarios explican la razón, nunca narran. No uses separadores con
  guion largo en comentarios ni documentación.
- Nunca edites archivos generados directamente. Edita el template y las reglas
  JSON y regenera:

  ```bash
  uv run tools/codegen/generate.py gen --all --keep-going --run-formatters
  ```

### Python

- Usa docstrings estilo Google y mantén el código formateado con la
  configuración de `ruff`.
- Ejecuta el tooling de Python con `uv`, nunca con `pip` ni `poetry` directos.

Los mensajes de commit siguen las reglas de [Commits](#commits), compartidas
por humanos y agentes.

## Commits

Seguimos [Conventional Commits](https://www.conventionalcommits.org/):

```text
<tipo>(<scope>): <descripción>

[cuerpo opcional]

[footer opcional]
```

Usa un tipo que describa la naturaleza del cambio. Esta guía no fija una lista
de tipos; sigue la convención que se usa actualmente en el repositorio. El scope
es opcional, una sola palabra en minúsculas y nunca una frase. El asunto tiene
un máximo de 72 caracteres y usa imperativo ("add", no "added"). El cuerpo tiene
un máximo de 80 caracteres por línea y nunca repite el asunto.

Elige el estilo según el tamaño del cambio:

- **Summarized**: una línea, para cambios triviales.
- **Normal**: asunto más un cuerpo de 2 a 4 líneas, para cambios moderados.
- **Dense**: asunto, cuerpo de 3 a 6 líneas y bullets, para cambios en varios
  archivos, features con varias partes o fixes con varias causas raíz. Los
  bullets empiezan con un verbo de acción: `Add`, `Remove`, `Update`,
  `Replace`, `Implement`, `Extract`, `Rename`, `Improve`, `Standardize`,
  `Refactor`, `Optimize`, `Guard`, `Wrap`, `Split` o `Align`. Un cambio por
  bullet.

Por defecto, un commit por archivo. Agrupa archivos solo cuando sus cambios son
estrictamente idénticos: pares CUDA/HIP espejados, una declaración con su
definición, o un template con su salida regenerada. Los archivos nuevos que
pertenecen a una feature van juntos. Si hay duda, separa.

Escribe cada commit pendiente en `commits.md` en la raíz del repo. Cada entrada
lista las rutas de archivos, una línea en blanco, la palabra `command:`, otra
línea en blanco y un bloque de texto fence con los comandos `git add` y
`git commit`. Redactar no ejecuta escrituras de git.

Aplica los commits solo con aprobación explícita. Ejecuta los comandos en el
orden del archivo, un commit por comando, y después informa del SHA y el asunto
de cada uno y verifica con `git log` y un `git status` limpio para archivos
tracked. Ante el primer fallo, para; nunca saltes, reordenes ni fusiones
commits para sortear el error.

Los merge commits se crean solo cuando se piden explícitamente. Verifica el
rango real con `git log` entre las puntas de las ramas y describe exactamente
ese rango. Un merge solo afecta a la rama en la que estás; el puntero de la rama
origen no se mueve.

Nunca hagas push, rebase, resolución de conflictos, `git add -A` ni edites
archivos generados directamente. Añade al staging exactamente los archivos
listados y deja intacto el resto del contenido sin seguimiento.

## Testing

### Tests Nativos

Los tests nativos son suites GTest que ejecuta CTest mediante un preset
`*-test-*`. Un test que falla, o no encontrar tests, falla la ejecución; los
presets fijan `noTestsAction` a `error`. Los presets con sanitizadores toman su
entorno de `suppr.txt` a través de los scripts. No amplíes ni quites un
`GTEST_SKIP` sin una razón ligada a la spec.

```bash
scripts/run-tests.sh cpu-test-release-linux -- -V
```

### Tests Python Legacy

```bash
uv run --frozen pytest -q tests
```

Limita la recolección a `tests`. Ejecutar pytest desde la raíz del repo también
recoge los tests del submodule de vcpkg, que no son parte de NovaNN.

### Check de Codegen

CI regenera las salidas y ejecuta `git diff --exit-code`. Un pull request que
cambie un template o una regla debe incluir los archivos regenerados.

## CI

| Workflow | Trigger | Check requerido |
| --- | --- | --- |
| `.github/workflows/ci.yml` | PR a `dev` o `main`, push a ambas, manual | `ci` |
| `.github/workflows/branch-policy.yml` | PR a `main` | `branch-origin` |
| `.github/workflows/gpu-sim.yml` | PR y push a `dev`/`main` con cambios relevantes, semanal, manual | no |
| `.github/workflows/nightly.yml` | semanal, manual | no |

El check `ci` es un job agregado sobre CPU Linux, ASan, UBSan, Windows, Python
y codegen. El mismo gate aplica a los pull requests a `dev` y a `main`. Todos los
jobs que configuran un preset instalan `uv` antes, porque el módulo CMake de
runtime lista las salidas de codegen mediante `uv run` durante la configuración.
El workflow de GPU simulada es informativo: los runners no tienen GPU y un
simulador no cubre tiempos ni comportamientos específicos de hardware.

Los jobs de CPU fijan `ubuntu-26.04` porque aporta GCC 15, CMake 4.5,
clang-format 20.1.8, Rust y Python 3.14.4 sin instalaciones adicionales. El
simulador de GPU se queda en `ubuntu-24.04`, que es lo que soporta su action de
setup. Windows fija `windows-2025`.

## Proceso de Pull Request

### Antes de Abrir el Pull Request

- [ ] Los presets nativos afectados pasan en local
- [ ] `uv run --frozen pytest -q tests` pasa si cambió código Python
- [ ] Los archivos generados están sincronizados con sus templates
- [ ] La documentación coincide con el comportamiento
- [ ] Los commits siguen las reglas de [Commits](#commits)
- [ ] El pull request apunta a `dev`

### Template de PR

```markdown
## Descripción

[Descripción clara del cambio]

## Tipo de Cambio

[Describe la naturaleza del cambio]

## Checklist

- [ ] Tests añadidos o actualizados
- [ ] Documentación actualizada
- [ ] Archivos generados regenerados cuando aplica
- [ ] CI pasa en local

## Testing

[Presets y comandos usados]

## Notas Adicionales

[Contexto relevante]
```

### Proceso de Review

1. Un mantenedor revisa el pull request a `dev`.
2. Pueden pedirse cambios o aclaraciones.
3. Con `ci` en verde y el cambio aprobado, se mergea a `dev`.
4. Un pull request de `dev` a `main` ejecuta el mismo gate más `branch-origin`.
5. Con aprobación y checks en verde, se mergea a `main`.

## Reportar Bugs

### Antes de Reportar

- Busca en los [issues existentes](https://github.com/JOSE-MDG/NovaNN/issues) para evitar duplicados.
- Comprueba que es un bug de NovaNN y no un problema del entorno.

### Template de Issue

````markdown
**Descripción del Bug**

[Descripción clara y concisa]

**Pasos para Reproducir**

1. Commit o versión usada
2. Comando o preset ejecutado
3. Error observado

**Comportamiento Esperado**

[Lo que esperabas que ocurriera]

**Comportamiento Actual**

[Lo que ocurrió realmente]

**Entorno**

- Commit de NovaNN: [SHA]
- SO y versión: [por ejemplo Ubuntu 26.04]
- Compilador: [por ejemplo GCC 15.2]
- Versiones de CMake y Ninja
- Backend y dispositivo: [CPU, CUDA o HIP]
- Dtype y shape

**Logs**
[Pega la salida relevante]
````

Para un bug de Python legacy, añade la versión de Python y la de NumPy.

## Proponer Features

```markdown
**¿Resuelve el feature un problema?**
[Describe el problema que tienes]

**Describe la Solución Propuesta**
[Cómo te gustaría que funcionara]

**Alternativas Consideradas**
[Otras soluciones que consideraste]

**Implementación Sugerida**
[Archivos o módulos que esperas cambiar]

**Contexto Adicional**
[Papers, referencias o ejemplos de otros frameworks]
```

## Preguntas Frecuentes

### ¿Puedo contribuir si soy principiante?

Sí. Los issues etiquetados como `good first issue` son un buen punto de partida.

### ¿En qué idioma escribo código y documentación?

Los identificadores y comentarios de código van en inglés. La documentación se
mantiene en español e inglés. Los issues y pull requests pueden estar en
cualquiera de los dos idiomas.

### ¿Puedo abrir un pull request directamente contra `main`?

No. El check `branch-origin` lo rechaza. Abre primero el pull request contra
`dev`.

### ¿Necesito tests para un cambio solo de documentación?

No. El código nuevo sí necesita tests.

### ¿Cómo se crean los commits cuando lo hace un agente?

El agente los redacta en `commits.md` y espera. Nada se stagea ni se aplica
hasta que lo apruebes explícitamente. Ver [Commits](#commits).

### ¿Cuánto tarda la review?

Normalmente entre uno y tres días. Si tarda más, puedes mencionarlo en el pull
request.

## Contacto

- **GitHub Issues**: para bugs y features
- **Email**: josepemlengineer@gmail.com
- **Discussions**: para preguntas generales y discusiones

---

Gracias por contribuir a NovaNN.
