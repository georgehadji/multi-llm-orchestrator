"""
Connectors — External system connectors
=====================================
Module for connecting to external systems like databases, APIs, and other services.

Pattern: Strategy
Async: Yes — for I/O-bound connection operations
Layer: L3 Agents

Usage:
    from orchestrator.connectors import ConnectorManager
    connector_manager = ConnectorManager()
    db_connector = connector_manager.register_db_connector(
        name="main_db",
        config={"host": "localhost", "port": 5432, "database": "mydb"}
    )
    result = await db_connector.query("SELECT * FROM users")
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger("orchestrator.connectors")


class BaseConnector(ABC):
    """Base class for all connectors."""

    def __init__(self, name: str, config: dict[str, Any]):
        self.name = name
        self.config = config
        self.connected = False

    @abstractmethod
    async def connect(self):
        """Establish connection to the external system."""
        pass

    @abstractmethod
    async def disconnect(self):
        """Close connection to the external system."""
        pass

    @abstractmethod
    async def test_connection(self) -> bool:
        """Test if the connection is working."""
        pass


class DatabaseConnector(BaseConnector):
    """Connector for database systems."""

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.connection = None
        self.driver = config.get("driver", "postgresql")  # Default to PostgreSQL

    async def connect(self):
        """Establish connection to the database."""
        try:
            if self.driver == "postgresql":
                import asyncpg
                self.connection = await asyncpg.connect(
                    host=self.config.get("host", "localhost"),
                    port=self.config.get("port", 5432),
                    user=self.config.get("user", "postgres"),
                    password=self.config.get("password", ""),
                    database=self.config.get("database", "postgres")
                )
            elif self.driver == "mysql":
                import aiomysql
                self.connection = await aiomysql.connect(
                    host=self.config.get("host", "localhost"),
                    port=self.config.get("port", 3306),
                    user=self.config.get("user", "root"),
                    password=self.config.get("password", ""),
                    db=self.config.get("database", "mysql")
                )
            elif self.driver == "sqlite":
                import aiosqlite
                self.connection = await aiosqlite.connect(
                    self.config.get("path", ":memory:")
                )
            else:
                raise ValueError(f"Unsupported database driver: {self.driver}")

            self.connected = True
            logger.info(f"Connected to database: {self.name}")
        except ImportError as e:
            logger.error(f"Database driver not installed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to database {self.name}: {e}")
            raise

    async def disconnect(self):
        """Close connection to the database."""
        if self.connection:
            if self.driver == "sqlite":
                await self.connection.close()
            else:
                self.connection.close()
                if hasattr(self.connection, 'wait_closed'):
                    await self.connection.wait_closed()

            self.connected = False
            logger.info(f"Disconnected from database: {self.name}")

    async def test_connection(self) -> bool:
        """Test if the database connection is working."""
        if not self.connected:
            return False

        try:
            if self.driver == "sqlite":
                cursor = await self.connection.execute("SELECT 1")
                await cursor.fetchone()
            else:
                await self.connection.fetchval("SELECT 1")
            return True
        except Exception:
            return False

    async def query(self, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
        """
        Execute a SELECT query.

        Args:
            sql: SQL query to execute
            params: Query parameters

        Returns:
            List of rows as dictionaries
        """
        if not self.connected:
            raise RuntimeError("Database not connected")

        try:
            if self.driver == "sqlite":
                if params:
                    cursor = await self.connection.execute(sql, params)
                else:
                    cursor = await self.connection.execute(sql)

                columns = [description[0] for description in cursor.description]
                rows = await cursor.fetchall()

                return [dict(zip(columns, row, strict=False)) for row in rows]
            else:
                rows = await self.connection.fetch(sql, *(params or ()))
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise

    async def execute(self, sql: str, params: list[Any] | None = None) -> int:
        """
        Execute an INSERT, UPDATE, or DELETE query.

        Args:
            sql: SQL query to execute
            params: Query parameters

        Returns:
            Number of affected rows
        """
        if not self.connected:
            raise RuntimeError("Database not connected")

        try:
            if self.driver == "sqlite":
                if params:
                    cursor = await self.connection.execute(sql, params)
                else:
                    cursor = await self.connection.execute(sql)

                await self.connection.commit()
                return cursor.rowcount
            else:
                result = await self.connection.execute(sql, *(params or ()))
                # Extract the number of affected rows from the result
                if "INSERT" in sql.upper():
                    return 1  # asyncpg returns the query string for INSERT
                return int(result.split()[-1]) if result.split() else 0
        except Exception as e:
            logger.error(f"Execute failed: {e}")
            raise


class HTTPConnector(BaseConnector):
    """Connector for HTTP APIs."""

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.session = None
        self.base_url = config.get("base_url", "")
        self.timeout = config.get("timeout", 30)
        self.headers = config.get("headers", {})

    async def connect(self):
        """Initialize the HTTP session."""
        import aiohttp
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(timeout=timeout, headers=self.headers)
        self.connected = True
        logger.info(f"HTTP connector initialized: {self.name}")

    async def disconnect(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.close()
        self.connected = False
        logger.info(f"HTTP connector closed: {self.name}")

    async def test_connection(self) -> bool:
        """Test if the HTTP connection is working by making a simple request."""
        if not self.connected:
            return False

        try:
            # Make a simple GET request to the base URL or a health endpoint
            url = f"{self.base_url}/health" if self.base_url else "https://httpbin.org/get"
            async with self.session.get(url) as response:
                return response.status == 200
        except Exception:
            return False

    async def get(self, endpoint: str, params: dict[str, Any] | None = None,
                  headers: dict[str, str] | None = None) -> dict[str, Any]:
        """
        Make a GET request.

        Args:
            endpoint: API endpoint to call
            params: Query parameters
            headers: Additional headers

        Returns:
            Response JSON as dictionary
        """
        if not self.connected:
            raise RuntimeError("HTTP connector not connected")

        url = f"{self.base_url}{endpoint}" if self.base_url else endpoint
        all_headers = {**self.headers, **(headers or {})}

        try:
            async with self.session.get(url, params=params, headers=all_headers) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"GET request failed: {e}")
            raise

    async def post(self, endpoint: str, data: dict[str, Any] | None = None,
                   json_data: dict[str, Any] | None = None,
                   headers: dict[str, str] | None = None) -> dict[str, Any]:
        """
        Make a POST request.

        Args:
            endpoint: API endpoint to call
            data: Form data
            json_data: JSON payload
            headers: Additional headers

        Returns:
            Response JSON as dictionary
        """
        if not self.connected:
            raise RuntimeError("HTTP connector not connected")

        url = f"{self.base_url}{endpoint}" if self.base_url else endpoint
        all_headers = {**self.headers, **(headers or {})}

        try:
            async with self.session.post(url, data=data, json=json_data, headers=all_headers) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"POST request failed: {e}")
            raise

    async def put(self, endpoint: str, data: dict[str, Any] | None = None,
                  json_data: dict[str, Any] | None = None,
                  headers: dict[str, str] | None = None) -> dict[str, Any]:
        """
        Make a PUT request.

        Args:
            endpoint: API endpoint to call
            data: Form data
            json_data: JSON payload
            headers: Additional headers

        Returns:
            Response JSON as dictionary
        """
        if not self.connected:
            raise RuntimeError("HTTP connector not connected")

        url = f"{self.base_url}{endpoint}" if self.base_url else endpoint
        all_headers = {**self.headers, **(headers or {})}

        try:
            async with self.session.put(url, data=data, json=json_data, headers=all_headers) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"PUT request failed: {e}")
            raise

    async def delete(self, endpoint: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
        """
        Make a DELETE request.

        Args:
            endpoint: API endpoint to call
            headers: Additional headers

        Returns:
            Response JSON as dictionary
        """
        if not self.connected:
            raise RuntimeError("HTTP connector not connected")

        url = f"{self.base_url}{endpoint}" if self.base_url else endpoint
        all_headers = {**self.headers, **(headers or {})}

        try:
            async with self.session.delete(url, headers=all_headers) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"DELETE request failed: {e}")
            raise


class FileConnector(BaseConnector):
    """Connector for file systems and cloud storage."""

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.storage_type = config.get("type", "local")  # local, s3, gcs, azure
        self.root_path = config.get("root_path", "./files")
        self.connected = True  # Local file system is always accessible

    async def connect(self):
        """Connect to the file storage (for remote storage types)."""
        if self.storage_type != "local":
            if self.storage_type == "s3":
                try:
                    import aioboto3
                    self.session = aioboto3.Session(
                        aws_access_key_id=self.config.get("aws_access_key_id"),
                        aws_secret_access_key=self.config.get("aws_secret_access_key"),
                        region_name=self.config.get("region", "us-east-1")
                    )
                    self.s3_client = await self.session.client("s3").__aenter__()
                except ImportError:
                    logger.error("aioboto3 not installed for S3 support")
                    raise
            elif self.storage_type == "gcs":
                try:
                    from gcloud.aio.storage import Storage
                    self.gcs_client = Storage(service_file=self.config.get("service_file"))
                except ImportError:
                    logger.error("gcloud-aio-storage not installed for GCS support")
                    raise
            elif self.storage_type == "azure":
                try:
                    from azure.storage.blob.aio import BlobServiceClient
                    self.azure_client = BlobServiceClient(
                        account_url=self.config.get("account_url"),
                        credential=self.config.get("credential")
                    )
                except ImportError:
                    logger.error("azure-storage-blob not installed for Azure support")
                    raise
        logger.info(f"File connector initialized: {self.name}")

    async def disconnect(self):
        """Disconnect from the file storage."""
        if self.storage_type == "s3" and hasattr(self, 's3_client'):
            await self.s3_client.__aexit__(None, None, None)
        elif self.storage_type == "azure" and hasattr(self, 'azure_client'):
            await self.azure_client.close()
        logger.info(f"File connector closed: {self.name}")

    async def test_connection(self) -> bool:
        """Test if the file storage connection is working."""
        try:
            if self.storage_type == "local":
                import os
                return os.path.exists(self.root_path)
            elif self.storage_type == "s3":
                await self.s3_client.list_buckets()
                return True
            elif self.storage_type == "gcs":
                await self.gcs_client.list_objects(bucket=self.config.get("bucket", ""))
                return True
            elif self.storage_type == "azure":
                containers = self.azure_client.list_containers()
                async for _ in containers:
                    return True
                return True  # If no containers exist, that's still valid
        except Exception:
            return False

    async def read_file(self, file_path: str) -> bytes:
        """
        Read a file from storage.

        Args:
            file_path: Path to the file

        Returns:
            File contents as bytes
        """
        if self.storage_type == "local":
            import os
            full_path = os.path.join(self.root_path, file_path.lstrip('/'))
            with open(full_path, 'rb') as f:
                return f.read()
        elif self.storage_type == "s3":
            obj = await self.s3_client.get_object(Bucket=self.config.get("bucket"), Key=file_path)
            return await obj['Body'].read()
        elif self.storage_type == "gcs":
            bucket = self.config.get("bucket", "")
            return await self.gcs_client.download_object(bucket=bucket, object_name=file_path)
        elif self.storage_type == "azure":
            blob_client = self.azure_client.get_blob_client(
                container=self.config.get("container", ""),
                blob=file_path
            )
            download_stream = await blob_client.download_blob()
            return await download_stream.readall()

    async def write_file(self, file_path: str, content: bytes) -> bool:
        """
        Write content to a file in storage.

        Args:
            file_path: Path to the file
            content: Content to write

        Returns:
            True if successful, False otherwise
        """
        if self.storage_type == "local":
            import os
            full_path = os.path.join(self.root_path, file_path.lstrip('/'))
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'wb') as f:
                f.write(content)
            return True
        elif self.storage_type == "s3":
            await self.s3_client.put_object(
                Bucket=self.config.get("bucket"),
                Key=file_path,
                Body=content
            )
            return True
        elif self.storage_type == "gcs":
            bucket = self.config.get("bucket", "")
            await self.gcs_client.upload_object(
                bucket=bucket,
                object_name=file_path,
                file_data=content
            )
            return True
        elif self.storage_type == "azure":
            blob_client = self.azure_client.get_blob_client(
                container=self.config.get("container", ""),
                blob=file_path
            )
            await blob_client.upload_blob(content, overwrite=True)
            return True

    async def list_files(self, prefix: str = "") -> list[str]:
        """
        List files in storage with an optional prefix.

        Args:
            prefix: Optional prefix to filter files

        Returns:
            List of file paths
        """
        if self.storage_type == "local":
            import os
            full_path = os.path.join(self.root_path, prefix.lstrip('/'))
            files = []
            for root, _, filenames in os.walk(full_path):
                for filename in filenames:
                    rel_path = os.path.relpath(os.path.join(root, filename), self.root_path)
                    files.append(rel_path.replace('\\', '/'))  # Normalize path separators
            return files
        elif self.storage_type == "s3":
            response = await self.s3_client.list_objects_v2(
                Bucket=self.config.get("bucket"),
                Prefix=prefix
            )
            return [obj['Key'] for obj in response.get('Contents', [])]
        elif self.storage_type == "gcs":
            bucket = self.config.get("bucket", "")
            objects = await self.gcs_client.list_objects(bucket=bucket, prefix=prefix)
            return [obj.name for obj in objects]
        elif self.storage_type == "azure":
            blob_list = self.azure_client.get_container_client(
                container=self.config.get("container", "")
            ).list_blobs(name_starts_with=prefix)
            files = []
            async for blob in blob_list:
                files.append(blob.name)
            return files


class ConnectorManager:
    """Manages different types of external system connectors."""

    def __init__(self):
        """Initialize the connector manager."""
        self.connectors: dict[str, BaseConnector] = {}

    def register_db_connector(self, name: str, config: dict[str, Any]) -> DatabaseConnector:
        """
        Register a database connector.

        Args:
            name: Name of the connector
            config: Configuration for the database connection

        Returns:
            DatabaseConnector instance
        """
        connector = DatabaseConnector(name, config)
        self.connectors[name] = connector
        logger.info(f"Registered database connector: {name}")
        return connector

    def register_http_connector(self, name: str, config: dict[str, Any]) -> HTTPConnector:
        """
        Register an HTTP connector.

        Args:
            name: Name of the connector
            config: Configuration for the HTTP connection

        Returns:
            HTTPConnector instance
        """
        connector = HTTPConnector(name, config)
        self.connectors[name] = connector
        logger.info(f"Registered HTTP connector: {name}")
        return connector

    def register_file_connector(self, name: str, config: dict[str, Any]) -> FileConnector:
        """
        Register a file connector.

        Args:
            name: Name of the connector
            config: Configuration for the file connection

        Returns:
            FileConnector instance
        """
        connector = FileConnector(name, config)
        self.connectors[name] = connector
        logger.info(f"Registered file connector: {name}")
        return connector

    async def get_connector(self, name: str) -> BaseConnector | None:
        """
        Get a connector by name.

        Args:
            name: Name of the connector

        Returns:
            Connector instance or None if not found
        """
        return self.connectors.get(name)

    async def connect_all(self):
        """Connect to all registered connectors."""
        for name, connector in self.connectors.items():
            try:
                await connector.connect()
            except Exception as e:
                logger.error(f"Failed to connect to {name}: {e}")

    async def disconnect_all(self):
        """Disconnect from all registered connectors."""
        for name, connector in self.connectors.items():
            try:
                await connector.disconnect()
            except Exception as e:
                logger.error(f"Failed to disconnect from {name}: {e}")

    async def test_all_connections(self) -> dict[str, bool]:
        """
        Test connections for all registered connectors.

        Returns:
            Dict mapping connector names to connection status
        """
        results = {}
        for name, connector in self.connectors.items():
            try:
                results[name] = await connector.test_connection()
            except Exception as e:
                logger.error(f"Failed to test connection for {name}: {e}")
                results[name] = False
        return results

    def get_connector_stats(self) -> dict[str, Any]:
        """
        Get statistics about registered connectors.

        Returns:
            Dict with connector statistics
        """
        stats = {
            "total_connectors": len(self.connectors),
            "connector_types": {},
            "connected_count": 0
        }

        for connector in self.connectors.values():
            conn_type = type(connector).__name__
            stats["connector_types"][conn_type] = stats["connector_types"].get(conn_type, 0) + 1
            if connector.connected:
                stats["connected_count"] += 1

        return stats
