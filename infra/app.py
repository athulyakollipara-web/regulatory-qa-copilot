#!/usr/bin/env python3
import aws_cdk as cdk

from stack import RegQaStack

app = cdk.App()
RegQaStack(app, "RegulatoryQaCopilot")
app.synth()
