#!/bin/bash
source /opt/google/c2d-utils

ANACONDA_PATH="/opt/conda/bin/jupyter"
if [[ "${ENABLE_MULTI_ENV,,}" == 'true' ]]; then
  ANACONDA_PATH="/opt/conda/envs/jupyterlab/bin/jupyter"
fi

#Configure gcloud for Dataproc Extension
set_cloud_sdk_config

# Check Dataproc extension logic for containers is now done in entry point.
disable_mixer=$(get_attribute_value disable-mixer)
if [[ ${disable_mixer,,} == "true" ]]; then
  "${ANACONDA_PATH}" server extension disable dataproc_jupyter_plugin
  "${ANACONDA_PATH}" labextension disable dataproc_jupyter_plugin
else
  "${ANACONDA_PATH}" server extension enable dataproc_jupyter_plugin
  "${ANACONDA_PATH}" labextension enable dataproc_jupyter_plugin
fi

# Toggle terminal
"${ANACONDA_PATH}" server extension enable jupyter_server_terminals

"${ANACONDA_PATH}" lab --allow-root --ip 0.0.0.0 --config=/opt/jupyter/.jupyter/jupyter_notebook_config.py