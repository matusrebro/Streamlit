mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
token = "$MAPBOX_TOKEN"\n\
\n\
" > ~/.streamlit/config.toml