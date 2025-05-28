import os

from alembic.config import Config
from alembic.runtime.environment import EnvironmentContext
from alembic.script import ScriptDirectory
from sqlalchemy import engine_from_config


def run_migrations(
    db_file_name: str,
):
    db_url = f"sqlite:///{db_file_name}"
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Create Alembic config object
    alembic_cfg = Config()
    alembic_cfg.set_main_option("script_location", current_dir)
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)

    # Create Script Directory object
    script = ScriptDirectory.from_config(alembic_cfg)

    # Create engine
    connectable = engine_from_config(
        alembic_cfg.get_section(alembic_cfg.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=None,
    )

    def fn_run_migration(rev, context):
        return script._upgrade_revs("head", rev)

    with connectable.connect() as connection:
        context = EnvironmentContext(
            config=alembic_cfg,
            script=script,
            fn=fn_run_migration,
        )

        context.configure(
            connection=connection,
            target_metadata=None,
            fn=fn_run_migration,
            transaction_per_migration=True,
        )

        with context.begin_transaction():
            context.run_migrations()
