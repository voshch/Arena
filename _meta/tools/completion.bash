# bash completion for the arena shell function. Sourced by _meta/tools/source.

_arena_complete() {
    local line="${COMP_LINE:0:COMP_POINT}"
    local cur="${line##*[[:space:]]}"
    local -a words=()
    read -r -a words <<< "$line"
    [ -z "$cur" ] && words+=("")

    local out
    out=$(python3 "$TOOLS_DIR/arena_cli/__main__.py" complete "${words[@]}" 2>/dev/null) || out=""

    local prefix="" nospace=0 files=0 head
    head="${cur%"${cur##*[$COMP_WORDBREAKS]}"}"
    COMPREPLY=()
    local entry value
    while IFS= read -r entry; do
        case "$entry" in
            "") ;;
            "@prefix="*) prefix="${entry#@prefix=}" ;;
            "@nospace") nospace=1 ;;
            "@files") files=1 ;;
            "@"*) ;;
            *)
                value="${entry%%$'\t'*}"
                value="$prefix$value"
                COMPREPLY+=("${value:${#head}}")
            ;;
        esac
    done <<< "$out"

    if [ "${#COMPREPLY[@]}" -gt 1 ]; then
        local -a shown=() v
        for v in "${COMPREPLY[@]}"; do shown+=("${v%:=}"); done
        COMPREPLY=("${shown[@]}")
    fi
    if [ "$files" = 1 ]; then
        compopt -o filenames +o nospace 2>/dev/null
        local f
        while IFS= read -r f; do COMPREPLY+=("$f"); done < <(compgen -f -- "${cur:${#head}}")
    elif [ "${#COMPREPLY[@]}" = 1 ] && [ "$nospace" != 1 ]; then
        COMPREPLY[0]="${COMPREPLY[0]} "
    fi
}

complete -o nospace -F _arena_complete arena
