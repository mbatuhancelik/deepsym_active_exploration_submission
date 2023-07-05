:- dynamic(experience/5).

bind_rest(ObjBinded, RelBinded, ObjPre, RelPre, ObjPost, RelPost) :-
    findall(
        PostValue,
        (
            nth0(Index, ObjPre, PreVal),
            (member(Index-BindedVal, ObjBinded) ->
                PostValue = BindedVal;
                PostValue = PreVal
            )
        ),
        ObjPost
    ),
    findall(
        RelPostValue,
        (
            nth0(RelIndex, RelPre, RelPreVal),
            findall(
                QueryPostValue,
                (
                    nth0(QueryIndex, RelPreVal, RelQueryPreVal),
                    findall(
                        QueryTargetPostValue,
                        (
                            nth0(TargetIndex, RelQueryPreVal, RelQueryTargetPreVal),
                            (member(RelIndex-QueryIndex-TargetIndex-RelBindedVal, RelBinded) ->
                                QueryTargetPostValue = RelBindedVal;
                                QueryTargetPostValue = RelQueryTargetPreVal
                            )
                        ),
                        QueryPostValue
                    )
                ),
                RelPostValue
            )
        ),
        RelPost
    ).

all_different([]).
all_different([H|T]) :-
    \+member(H, T),
    all_different(T).

dummy(A, B, C, D, E) :- experience(A, B, C, D, E), !.