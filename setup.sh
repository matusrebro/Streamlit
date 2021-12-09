mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
[mapbox]\n\
token = \"$MAPBOX_TOKEN\"\n\
\n\
" > ~/.streamlit/config.toml