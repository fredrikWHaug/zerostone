# Zerostone

### Contributing

1. **Fork the repository** and clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/zerostone.git
   cd zerostone
   ```

2. **Create a feature branch** with the `feature/` prefix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes** and test locally.

4. **Run CI checks** before committing:
   ```bash
   ./ci-local.sh
   ```
   This runs the exact same checks as GitHub CI/CD:
   - Code formatting (`cargo fmt`)
   - Linting (`cargo clippy`)
   - Documentation (`cargo doc`)
   - Compilation (`cargo check`)
   - Tests (`cargo test`)

5. **Commit and push** your changes:
   ```bash
   git add .
   git commit -m "Your commit message"
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request** from your fork's feature branch to the main repository.
