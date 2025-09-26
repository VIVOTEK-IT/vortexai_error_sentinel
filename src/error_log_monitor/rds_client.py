"""
RDS Database Client for Data Integrity Checks.

This module provides read-only access to the Vortexai RDS database
for checking data integrity during error analysis.
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, Optional, Any
import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)


@dataclass
class RDSConfig:
    """RDS database configuration."""

    host: str
    port: int
    database: str
    username: str
    password: str
    ssl_mode: str = "require"


class RDSClient:
    """Read-only RDS client for data integrity checks."""

    def __init__(self, config: RDSConfig):
        """Initialize RDS client with configuration."""
        self.config = config
        self.connection = None

    def connect(self) -> bool:
        """Connect to RDS database."""
        try:
            self.connection = psycopg.connect(
                host=self.config.host,
                port=self.config.port,
                dbname=self.config.database,
                user=self.config.username,
                password=self.config.password,
                sslmode=self.config.ssl_mode,
                row_factory=dict_row,
            )
            logger.info("Successfully connected to RDS database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to RDS database: {e}", exc_info=True)
            return False

    def disconnect(self):
        """Disconnect from RDS database."""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Disconnected from RDS database")

    def get_camera_info(self, mac: str) -> Dict[str, Any]:
        """
        Get CAMERA_INFO table for a specific camera.

        Args:
            mac: Camera MAC address

        Returns:
            a row of CAMERA_INFO table
        """
        if not self.connection:
            logger.warning("Not connected to database")
            return {"error": "Not connected to database"}

        try:
            with self.connection.cursor() as cursor:
                # Check if camera exists and get basic info
                cursor.execute(
                    """
                    SELECT mac, thingname, last_report_time, group_id, 
                           organization_id, partition_group_id, license_period_in_day
                    FROM CAMERA_INFO 
                    WHERE mac = %s
                """,
                    (mac),
                )

                result = cursor.fetchone()

                if not result:
                    return None
                return {
                    "mac": result[0],
                    "thingname": result[1],
                    "last_report_time": result[2],
                    "group_id": result[3],
                    "organization_id": result[4],
                    "partition_group_id": result[5],
                    "license_period_in_day": result[6],
                }

        except Exception as e:
            logger.warning(f"Error checking CAMERA_INFO integrity: {e}")
            return None

    def check_object_trace_partition_exists(self, partition_id: str) -> Dict[str, Any]:
        try:
            with self.connection.cursor() as cursor:
                # Check if partition table exists
                cursor.execute(
                    """
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables 
                        WHERE table_name = %s
                    )
                """,
                    (f"object_trace_{partition_id}",),
                )

                partition_exists = cursor.fetchone()[0]

                if not partition_exists:
                    return False
            return True
        except Exception as e:
            logger.warning(f"Error checking OBJECT_TRACE partition existence: {e}")
        return False

    def get_partition_info(self, mac, partition_id: str) -> dict[str, Any]:
        """
        Get PARTITION_INFO table for a specific mac and partition_id.

        Args:
            partition_id: Partition Group ID
        """
        if not self.connection:
            return None
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    "SELECT partition_id, mac, tbl_name_object_trace, last_update FROM partition_info WHERE mac = %s AND partition_id = %s;"(
                        mac, partition_id
                    ),
                )
                result = cursor.fetchone()
                if not result:
                    return None

                return {
                    "partition_id": result[0],
                    "mac": result[1],
                    "tbl_name_object_trace": result[2],
                    "last_update": result[3],
                }
        except Exception as e:
            logger.warning(f"Error checking PARTITION_INFO integrity: {e}")
            return None

    def get_object_trace(self, mac: str, partition_id: str) -> dict[str, Any]:
        """
        Check OBJECT_TRACE table integrity for a specific camera and partition.

        Args:
            mac: Camera MAC address
            partition_id: Partition ID

        Returns:
            Dictionary with integrity check results
        """
        if not self.connection:
            return {"error": "Not connected to database"}
        if not self.check_object_trace_partition_exists(partition_id):
            return None

        try:
            with self.connection.cursor() as cursor:
                # Check data integrity for the specific mac and partition
                cursor.execute(
                    """
                    SELECT oid, mac, obj_type, partition_id, first_time, invalidate
                    FROM OBJECT_TRACE 
                    WHERE mac = %s AND partition_id = %s
                    LIMIT 1;
                """,
                    (mac, partition_id),
                )

                result = cursor.fetchone()
                return {
                    "oid": result[0],
                    "mac": result[1],
                    "obj_type": result[2],
                    "partition_id": result[3],
                    "first_time": result[4],
                    "invalidate": result[5],
                }

        except Exception as e:
            logger.error(f"Error checking OBJECT_TRACE integrity: {e}")
            return None

    # def check_data_integrity_for_error(self, error_message: str, service: str) -> Dict[str, Any]:
    #     """
    #     Check data integrity based on error context.

    #     Args:
    #         error_message: Error message to analyze
    #         service: Service name (e.g., "UPDATE_CAMERAINFO", "PARSE_METADATA")

    #     Returns:
    #         Dictionary with data integrity assessment
    #     """
    #     if not self.connection:
    #         return {"error": "Not connected to database"}

    #     # Extract potential identifiers from error message
    #     mac = self._extract_mac_from_error(error_message)
    #     partition_id = self._extract_partition_from_error(error_message)

    #     results = {
    #         "service": service,
    #         "error_message": error_message,
    #         "data_integrity_status": "UNKNOWN",
    #         "checks_performed": [],
    #         "recommendations": [],
    #     }

    #     if service == "UPDATE_CAMERAINFO" and mac:
    #         # Check CAMERA_INFO table
    #         camera_check = self.check_camera_info_integrity(mac, partition_id or -1)
    #         results["checks_performed"].append("CAMERA_INFO")
    #         results["camera_info_check"] = camera_check

    #         if camera_check.get("integrity_status") == "DEGRADED":
    #             results["data_integrity_status"] = "DAMAGED"
    #             results["recommendations"].append("Check CAMERA_INFO table for data corruption")
    #         elif camera_check.get("integrity_status") == "HEALTHY":
    #             results["data_integrity_status"] = "INTACT"

    #     elif service == "PARSE_METADATA" and mac and partition_id:
    #         # Check OBJECT_TRACE table
    #         trace_check = self.check_object_trace_integrity(mac, partition_id)
    #         results["checks_performed"].append("OBJECT_TRACE")
    #         results["object_trace_check"] = trace_check

    #         if trace_check.get("integrity_status") == "DEGRADED":
    #             results["data_integrity_status"] = "DAMAGED"
    #             results["recommendations"].append("Check OBJECT_TRACE table for data corruption")
    #         elif trace_check.get("integrity_status") == "HEALTHY":
    #             results["data_integrity_status"] = "INTACT"

    #     else:
    #         results["data_integrity_status"] = "UNABLE_TO_CHECK"
    #         results["recommendations"].append("Unable to extract identifiers for data integrity check")

    #     return results

    def _extract_mac_from_error(self, error_message: str) -> Optional[str]:
        """Extract MAC address from error message."""
        # Look for MAC address patterns (e.g., "mac-1234567890" or similar)
        mac_pattern = r'mac[_-]?([a-fA-F0-9]{6,})'
        match = re.search(mac_pattern, error_message)
        return match.group(1) if match else None

    def _extract_partition_from_error(self, error_message: str) -> Optional[str]:
        """Extract partition ID from error message."""
        # Look for partition patterns
        partition_pattern = r'partition[_-]?(\d+)'
        match = re.search(partition_pattern, error_message)
        return match.group(1) if match else None


def create_rds_client(config: RDSConfig) -> Optional[RDSClient]:
    """Create RDS client from configuration."""
    try:
        if not all([config.host, config.database, config.username, config.password]):
            logger.warning("RDS configuration incomplete, skipping RDS integration")
            return None

        client = RDSClient(config)
        if client.connect():
            return client
        else:
            return None

    except Exception as e:
        logger.error(f"Failed to create RDS client: {e}")
        return None
