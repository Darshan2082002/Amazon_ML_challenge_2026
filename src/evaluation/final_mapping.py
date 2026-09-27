import os
import pandas as pd


class UnionFind:

    def __init__(self):
        self.parent = {}

    def add(self, x):

        if x not in self.parent:
            self.parent[x] = x

    def find(self, x):

        if self.parent[x] != x:
            self.parent[x] = self.find(
                self.parent[x]
            )

        return self.parent[x]

    def union(self, a, b):

        self.add(a)
        self.add(b)

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a != root_b:
            self.parent[root_b] = root_a


def create_final_mapping(
    matching_results_path,
    output_directory
):

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    df = pd.read_csv(
        matching_results_path,
        sep="\t"
    )

    print(
        "\nMatching results columns:",
        list(df.columns)
    )

    source_col = "source1_entity_id"

    candidate_col = "matched_entity_ids"

    if source_col not in df.columns:
        raise ValueError(
            f"Missing column: {source_col}"
        )

    if candidate_col not in df.columns:
        raise ValueError(
            f"Missing column: {candidate_col}"
        )

    uf = UnionFind()

    # Process each S1 → candidate relationship
    for _, row in df.iterrows():

        s1_id = str(
            row[source_col]
        ).strip()

        matched = row[candidate_col]

        if pd.isna(matched):
            continue

        matched = str(
            matched
        ).strip()

        if not matched:
            continue

        s1_node = f"S1::{s1_id}"

        uf.add(s1_node)

        candidate_ids = [
            x.strip()
            for x in matched.split(",")
            if x.strip()
        ]

        for candidate_id in candidate_ids:

            # Detect source from ID
            if candidate_id.startswith("S2"):
                node = f"S2::{candidate_id}"

            elif candidate_id.startswith("S3"):
                node = f"S3::{candidate_id}"

            else:
                node = f"OTHER::{candidate_id}"

            uf.union(
                s1_node,
                node
            )

    # Group nodes
    groups = {}

    for node in uf.parent:

        root = uf.find(node)

        groups.setdefault(
            root,
            []
        ).append(node)

    rows = []

    entity_number = 1

    for _, members in groups.items():

        entity_id = (
            f"E{entity_number:06d}"
        )

        entity_number += 1

        for member in members:

            source, original_id = (
                member.split(
                    "::",
                    1
                )
            )

            rows.append({
                "entity_id": entity_id,
                "source": source,
                "original_entity_id": original_id
            })

    final_df = pd.DataFrame(rows)

    output_path = os.path.join(
        output_directory,
        "final_entity_mapping.csv"
    )

    final_df.to_csv(
        output_path,
        index=False
    )

    print(
        "\nFinal entity mapping saved to:",
        output_path
    )

    return final_df