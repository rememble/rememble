"""Knowledge graph search: entity/observation/relation queries."""

from __future__ import annotations

import sqlite3

from rememble.models import Entity, GraphResult, Observation, Relation, RelationWithEntity


def graphSearch(
    db: sqlite3.Connection,
    query: str,
    limit: int = 10,
    project: str | None = None,
) -> list[GraphResult]:
    """Search knowledge graph by entity name or observation content.

    1. Fuzzy match entities by name
    2. Fuzzy match observations by content → resolve to entities
    3. Collect observations + 1-hop relations for each matched entity
    """
    pattern = f"%{query}%"

    project_clause = ""
    project_params: list[str] = []
    if project is not None:
        project_clause = " AND (e.project = ? OR e.project IS NULL)"
        project_params = [project]

    # Find entities matching by name
    entity_rows = db.execute(
        f"""SELECT DISTINCT e.* FROM entities e
           WHERE e.name LIKE ? COLLATE NOCASE{project_clause}
           LIMIT ?""",
        (pattern, *project_params, limit),
    ).fetchall()

    # Find entities matching by observation content
    obs_entity_rows = db.execute(
        f"""SELECT DISTINCT e.* FROM entities e
           JOIN observations o ON o.entity_id = e.id
           WHERE o.content LIKE ? COLLATE NOCASE{project_clause}
           LIMIT ?""",
        (pattern, *project_params, limit),
    ).fetchall()

    # Deduplicate entities
    seen_ids: set[int] = set()
    entities: list[sqlite3.Row] = []
    for row in [*entity_rows, *obs_entity_rows]:
        if row["id"] not in seen_ids:
            seen_ids.add(row["id"])
            entities.append(row)

    results: list[GraphResult] = []
    for e_row in entities[:limit]:
        entity = Entity(
            id=e_row["id"],
            name=e_row["name"],
            entity_type=e_row["entity_type"],
            created_at=e_row["created_at"],
            updated_at=e_row["updated_at"],
        )

        # Get observations
        obs_rows = db.execute(
            "SELECT * FROM observations WHERE entity_id = ? ORDER BY created_at DESC",
            (entity.id,),
        ).fetchall()
        observations = [
            Observation(
                id=o["id"],
                entity_id=o["entity_id"],
                content=o["content"],
                source=o["source"],
                valid_from=o["valid_from"],
                valid_to=o["valid_to"],
                created_at=o["created_at"],
            )
            for o in obs_rows
        ]

        # Get 1-hop relations (outbound + inbound)
        relations_with: list[RelationWithEntity] = []

        # Outbound
        out_rows = db.execute(
            """SELECT r.*, e.id AS e_id, e.name AS e_name, e.entity_type AS e_type,
                      e.created_at AS e_created, e.updated_at AS e_updated
               FROM relations r JOIN entities e ON e.id = r.to_entity_id
               WHERE r.from_entity_id = ?""",
            (entity.id,),
        ).fetchall()
        for r in out_rows:
            relations_with.append(
                RelationWithEntity(
                    relation=Relation(
                        id=r["id"],
                        from_entity_id=r["from_entity_id"],
                        to_entity_id=r["to_entity_id"],
                        relation_type=r["relation_type"],
                        metadata_json=r["metadata_json"],
                        created_at=r["created_at"],
                    ),
                    entity=Entity(
                        id=r["e_id"],
                        name=r["e_name"],
                        entity_type=r["e_type"],
                        created_at=r["e_created"],
                        updated_at=r["e_updated"],
                    ),
                    direction="outbound",
                )
            )

        # Inbound
        in_rows = db.execute(
            """SELECT r.*, e.id AS e_id, e.name AS e_name, e.entity_type AS e_type,
                      e.created_at AS e_created, e.updated_at AS e_updated
               FROM relations r JOIN entities e ON e.id = r.from_entity_id
               WHERE r.to_entity_id = ?""",
            (entity.id,),
        ).fetchall()
        for r in in_rows:
            relations_with.append(
                RelationWithEntity(
                    relation=Relation(
                        id=r["id"],
                        from_entity_id=r["from_entity_id"],
                        to_entity_id=r["to_entity_id"],
                        relation_type=r["relation_type"],
                        metadata_json=r["metadata_json"],
                        created_at=r["created_at"],
                    ),
                    entity=Entity(
                        id=r["e_id"],
                        name=r["e_name"],
                        entity_type=r["e_type"],
                        created_at=r["e_created"],
                        updated_at=r["e_updated"],
                    ),
                    direction="inbound",
                )
            )

        results.append(
            GraphResult(
                entity=entity,
                observations=observations,
                relations=relations_with,
            )
        )

    return results
