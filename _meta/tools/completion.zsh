# zsh completion for the arena shell function. Sourced by _meta/tools/source.

_arena() {
    local -a lines
    lines=("${(@f)$(python3 "$TOOLS_DIR/arena_cli/__main__.py" complete "${(@)words[1,CURRENT]}" 2>/dev/null)}")

    local prefix='' nospace=0 files=0 group='arena' line value desc shown
    local -a order
    local -A groups matches
    for line in "${lines[@]}"; do
        case "$line" in
            '') ;;
            '@prefix='*) prefix="${line#@prefix=}" ;;
            '@nospace') nospace=1 ;;
            '@files') files=1 ;;
            '@group='*)
                group="${line#@group=}"
                (( ${+groups[$group]} )) || { groups[$group]=''; matches[$group]=''; order+=("$group"); }
            ;;
            '@'*) ;;
            *)
                (( ${+groups[$group]} )) || { groups[$group]=''; matches[$group]=''; order+=("$group"); }
                # value<TAB>desc -> shown as value without :=, inserted as value
                value="${line%%$'\t'*}" desc="${line#*$'\t'}"
                [ "$desc" = "$line" ] && desc=''
                shown="${value%:=}"
                groups[$group]+="${shown//:/\\:}${desc:+:$desc}"$'\n'
                matches[$group]+="${value//:/\\:}"$'\n'
            ;;
        esac
    done

    [ -n "$prefix" ] && compset -P '*:='

    local -a opts=()
    (( nospace )) && opts=(-S '')
    local ret=1 tag
    local -a entries inserts
    for group in "${order[@]}"; do
        entries=("${(@f)groups[$group]}")
        entries=("${(@)entries:#}")
        inserts=("${(@f)matches[$group]}")
        inserts=("${(@)inserts:#}")
        tag="${${group:l}// /-}"
        _describe -t "arena-$tag" "$group" entries inserts "${opts[@]}" && ret=0
    done
    (( files )) && { _files && ret=0; }
    return ret
}

_arena_register_completion() {
    (( ${+functions[compdef]} )) || return 1
    compdef _arena arena
    return 0
}

if ! _arena_register_completion; then
    _arena_deferred_compdef() {
        _arena_register_completion && precmd_functions=("${(@)precmd_functions:#_arena_deferred_compdef}")
    }
    precmd_functions+=(_arena_deferred_compdef)
fi
