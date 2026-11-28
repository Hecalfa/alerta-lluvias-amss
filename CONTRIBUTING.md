# Cómo contribuir

1. Nunca subas cambios directo a `main`.
2. Crea una rama desde `main` actualizado:
   ```bash
   git checkout main
   git pull
   git checkout -b feature/nombre-corto
   ```
3. Haz commits pequeños con mensajes claros.
4. Sube tu rama y abre un Pull Request:
   ```bash
   git push -u origin feature/nombre-corto
   ```
5. Llena la plantilla del PR. QA (@cotan95) revisará y aprobará o pedirá cambios.
6. Cuando el PR esté aprobado, se hace merge a `main`.

Prefijos de rama: `feature/`, `fix/`, `data/`, `model/`, `docs/`.
