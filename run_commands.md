## Running a single test file
1. Activate the virtual environment
    PowerShell (Default Windows Terminal):
    ```powershell
    Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
    .venv\Scripts\activate.ps1
    ```

2. Run the file as a Python module from the root folder `ActCheck/`:
    ```powershell
    python -m tests.test_binary
    ```

3. Deactivate the virtual environment:
    ```powershell
    deactivate
    ```

### Running using UV
Just run this from root folder `ActCheck/`:

```powershell
uv run python -m tests.test_binary
```

Running any test is just running the Python file as a module, hence the `-m`. The `__init__.py` file existing in the `tests/` folder usually works for most errors.