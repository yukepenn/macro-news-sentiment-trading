# Contributing Guidelines

## Commit Message Convention

We follow a specific commit message format to keep our repository history clean and meaningful. This helps in:
- Filtering out less important commits from the main page
- Making it easier to understand the purpose of each change
- Maintaining a clean and professional repository history

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: A new feature (shows on main page)
- `fix`: A bug fix (shows on main page)
- `docs`: Documentation changes (shows on main page)
- `style`: Changes that do not affect the meaning of the code (hidden from main page)
- `refactor`: Code changes that neither fix a bug nor add a feature (hidden from main page)
- `perf`: Performance improvements (shows on main page)
- `test`: Adding or fixing tests (hidden from main page)
- `chore`: Changes to the build process or auxiliary tools (hidden from main page)
- `ci`: Changes to CI configuration files and scripts (hidden from main page)

### Scopes

- `api`: API related changes
- `ui`: User interface changes
- `deps`: Dependency updates
- `git`: Git related changes
- `config`: Configuration changes
- `utils`: Utility functions
- `model`: Model related changes
- `data`: Data processing changes
- `docs`: Documentation changes

### Examples

```bash
# Shows on main page
git commit -m "feat(model): add new sentiment analysis model"
git commit -m "fix(api): resolve API rate limiting issue"
git commit -m "docs(readme): update installation instructions"

# Hidden from main page
git commit -m "chore(deps): update dependencies"
git commit -m "style(ui): format code according to style guide"
git commit -m "test(unit): add unit tests for sentiment analysis"
```

### Important Notes

1. Use `chore`, `style`, `test`, `refactor`, and `ci` types for changes that don't need to be highlighted on the main page
2. Keep the subject line under 50 characters
3. Use imperative mood in the subject line
4. Separate subject from body with a blank line
5. Wrap the body at 72 characters
6. Use the footer to reference any related issues 