"""AWS CDK stack: corpus bucket, Bedrock Guardrail (prompt-attack + PII + contextual grounding),
container Lambda behind an HTTP API with Cognito JWT auth, least-privilege IAM."""
from __future__ import annotations

from pathlib import Path

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_apigatewayv2 as apigw
from aws_cdk import aws_apigatewayv2_authorizers as authorizers
from aws_cdk import aws_apigatewayv2_integrations as integrations
from aws_cdk import aws_bedrock as bedrock
from aws_cdk import aws_cognito as cognito
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_s3 as s3
from constructs import Construct

ROOT = Path(__file__).resolve().parents[1]
BLOCKED_SENTINEL = "BLOCKED_BY_GUARDRAIL"  # must match regqa/safety.py


class RegQaStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        bucket = s3.Bucket(self, "Corpus", encryption=s3.BucketEncryption.S3_MANAGED,
                           block_public_access=s3.BlockPublicAccess.BLOCK_ALL, enforce_ssl=True, versioned=True,
                           removal_policy=RemovalPolicy.DESTROY, auto_delete_objects=True)

        guardrail = bedrock.CfnGuardrail(
            self, "Guardrail",
            name="regqa",
            blocked_input_messaging=BLOCKED_SENTINEL,
            blocked_outputs_messaging=BLOCKED_SENTINEL,
            content_policy_config=bedrock.CfnGuardrail.ContentPolicyConfigProperty(filters_config=[
                bedrock.CfnGuardrail.ContentFilterConfigProperty(type="PROMPT_ATTACK", input_strength="HIGH", output_strength="NONE"),
            ]),
            sensitive_information_policy_config=bedrock.CfnGuardrail.SensitiveInformationPolicyConfigProperty(pii_entities_config=[
                bedrock.CfnGuardrail.PiiEntityConfigProperty(type=t, action="ANONYMIZE")
                for t in ("US_SOCIAL_SECURITY_NUMBER", "US_BANK_ACCOUNT_NUMBER", "EMAIL", "PHONE", "CREDIT_DEBIT_CARD_NUMBER")
            ]),
            # Answers that are not supported by the retrieved paragraphs, or not relevant to the question, are intercepted.
            contextual_grounding_policy_config=bedrock.CfnGuardrail.ContextualGroundingPolicyConfigProperty(filters_config=[
                bedrock.CfnGuardrail.ContextualGroundingFilterConfigProperty(type="GROUNDING", threshold=0.75),
                bedrock.CfnGuardrail.ContextualGroundingFilterConfigProperty(type="RELEVANCE", threshold=0.5),
            ]),
        )

        pool = cognito.UserPool(self, "Users", self_sign_up_enabled=False, sign_in_aliases=cognito.SignInAliases(email=True),
                                removal_policy=RemovalPolicy.DESTROY)
        client = pool.add_client("ApiClient", auth_flows=cognito.AuthFlow(user_password=True, user_srp=True))

        fn = lambda_.DockerImageFunction(
            self, "QaFn",
            code=lambda_.DockerImageCode.from_image_asset(str(ROOT), file="Dockerfile.lambda",
                                                          exclude=["data", "mlruns", "mlflow.db", "cdk.out", ".venv", ".git", "eval/results"]),
            memory_size=2048, timeout=Duration.seconds(60),
            environment={
                "REGQA_LLM": "bedrock", "REGQA_RERANKER": "bedrock", "REGQA_EMBEDDINGS": "titan",
                "REGQA_CORPUS": "/tmp/corpus.jsonl", "CORPUS_S3_URI": f"s3://{bucket.bucket_name}/corpus.jsonl",
                "BEDROCK_GUARDRAIL_ID": guardrail.attr_guardrail_id, "BEDROCK_GUARDRAIL_VERSION": "DRAFT",
            },
        )
        bucket.grant_read(fn, "corpus.jsonl")
        fn.add_to_role_policy(iam.PolicyStatement(
            actions=["bedrock:InvokeModel", "bedrock:Converse", "bedrock:Rerank"],
            resources=["arn:aws:bedrock:*::foundation-model/*", f"arn:aws:bedrock:*:{self.account}:inference-profile/*"],
        ))
        fn.add_to_role_policy(iam.PolicyStatement(actions=["bedrock:ApplyGuardrail"], resources=[guardrail.attr_guardrail_arn]))

        api = apigw.HttpApi(self, "Api", api_name="regqa")
        auth = authorizers.HttpUserPoolAuthorizer("CognitoAuth", pool, user_pool_clients=[client])
        integ = integrations.HttpLambdaIntegration("QaIntegration", fn)
        api.add_routes(path="/ask", methods=[apigw.HttpMethod.POST], integration=integ, authorizer=auth)
        api.add_routes(path="/health", methods=[apigw.HttpMethod.GET], integration=integ)

        CfnOutput(self, "ApiUrl", value=api.api_endpoint)
        CfnOutput(self, "CorpusBucket", value=bucket.bucket_name)
        CfnOutput(self, "GuardrailId", value=guardrail.attr_guardrail_id)
        CfnOutput(self, "UserPoolId", value=pool.user_pool_id)
        CfnOutput(self, "UserPoolClientId", value=client.user_pool_client_id)
