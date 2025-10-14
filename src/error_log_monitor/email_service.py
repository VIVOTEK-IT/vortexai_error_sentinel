"""AWS SES Email Service for sending daily reports."""

import logging
from datetime import datetime
from typing import List, Optional

import boto3
from botocore.exceptions import ClientError

from error_log_monitor.config import EmailConfig

logger = logging.getLogger(__name__)


class EmailService:
    """Service for sending emails via AWS SES."""

    def __init__(self, config: EmailConfig):
        self.config = config
        self.ses_client = boto3.client('ses', region_name=config.aws_region)

    def send_daily_report_email(
        self,
        html_content: str,
        subject: Optional[str] = None,
        recipients: Optional[List[str]] = None
    ) -> bool:
        """
        Send daily report email via AWS SES.

        Args:
            html_content: HTML content of the email
            subject: Email subject (defaults to formatted daily report subject)
            recipients: List of email recipients (defaults to config recipients)

        Returns:
            True if email was sent successfully, False otherwise
        """
        if not recipients:
            recipients = self.config.recipients

        if not recipients:
            logger.error("No email recipients configured")
            return False

        if not subject:
            today = datetime.now().strftime("%Y-%m-%d")
            subject = f"[{today}][Vortexai Error Issue] daily issue report"

        try:
            response = self.ses_client.send_email(
                Source=self.config.sender_email,
                Destination={
                    'ToAddresses': recipients
                },
                Message={
                    'Subject': {
                        'Data': subject,
                        'Charset': 'UTF-8'
                    },
                    'Body': {
                        'Html': {
                            'Data': html_content,
                            'Charset': 'UTF-8'
                        }
                    }
                }
            )
            logger.info(f"Daily report email sent successfully. MessageId: {response['MessageId']}, to {recipients}")
            return True

        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"Failed to send daily report email. Error: {error_code} - {error_message}")
            return False

        except Exception as e:
            logger.error(f"Unexpected error sending daily report email: {str(e)}")
            return False

    def verify_email_identity(self, email: str) -> bool:
        """
        Verify an email address with AWS SES.

        Args:
            email: Email address to verify

        Returns:
            True if verification request was successful, False otherwise
        """
        try:
            self.ses_client.verify_email_identity(EmailAddress=email)
            logger.info(f"Verification email sent to {email}")
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"Failed to verify email {email}. Error: {error_code} - {error_message}")
            return False

    def get_send_quota(self) -> dict:
        """
        Get current SES sending quota information.

        Returns:
            Dictionary containing quota information
        """
        try:
            response = self.ses_client.get_send_quota()
            return response
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"Failed to get send quota. Error: {error_code} - {error_message}")
            return {}

    def is_email_verified(self, email: str) -> bool:
        """
        Check if an email address is verified in SES.

        Args:
            email: Email address to check

        Returns:
            True if email is verified, False otherwise
        """
        try:
            response = self.ses_client.get_identity_verification_attributes(
                Identities=[email]
            )
            verification_attributes = response.get('VerificationAttributes', {})
            email_attributes = verification_attributes.get(email, {})
            return email_attributes.get('VerificationStatus') == 'Success'
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"Failed to check email verification status for {email}. Error: {error_code} - {error_message}")
            return False

