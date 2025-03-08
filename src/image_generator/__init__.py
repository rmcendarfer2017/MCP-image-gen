from . import server
import asyncio
import sys

def main():
    """Main entry point for the package."""
    try:
        asyncio.run(server.main())
    except KeyboardInterrupt:
        print("Server stopped by user.", file=sys.stderr)
    except Exception as e:
        import traceback
        print(f"Error running server: {str(e)}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)

# Optionally expose other important items at package level
__all__ = ['main', 'server']