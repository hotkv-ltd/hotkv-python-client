# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Optional integrations with third party AI frameworks.

Each submodule (`hotkv.integrations.langchain`, `hotkv.integrations.llama_index`,
`hotkv.integrations.litellm`) depends on the matching framework, installed
through an extra (`pip install "hotkv[langchain]"`, `"hotkv[llamaindex]"`,
`"hotkv[litellm]"`). None of them are imported here and none of them are
hard dependencies of the base `hotkv` package: importing `hotkv` itself
never requires any of these frameworks to be installed.
"""

from __future__ import annotations
