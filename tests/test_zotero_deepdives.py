import json
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import scripts.zotero_deepdives as zd


def _create_minimal_zotero_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    with conn:
        conn.executescript(
            """
            create table collections (
                collectionID integer primary key autoincrement,
                collectionName text,
                parentCollectionID integer,
                clientDateModified text,
                libraryID integer,
                key text unique,
                version integer,
                synced integer
            );
            create table items (
                itemID integer primary key autoincrement,
                itemTypeID integer,
                dateAdded text,
                dateModified text,
                clientDateModified text,
                libraryID integer,
                key text unique,
                version integer,
                synced integer
            );
            create table itemDataValues (
                valueID integer primary key autoincrement,
                value text unique
            );
            create table itemData (
                itemID integer,
                fieldID integer,
                valueID integer,
                primary key (itemID, fieldID)
            );
            create table creators (
                creatorID integer primary key autoincrement,
                firstName text,
                lastName text,
                fieldMode integer
            );
            create table itemCreators (
                itemID integer,
                creatorID integer,
                creatorTypeID integer,
                orderIndex integer,
                unique (itemID, creatorID, creatorTypeID, orderIndex)
            );
            create table itemAttachments (
                itemID integer,
                parentItemID integer,
                linkMode integer,
                contentType text,
                charsetID integer,
                path text,
                syncState integer
            );
            create table tags (
                tagID integer primary key autoincrement,
                name text unique
            );
            create table itemTags (
                itemID integer,
                tagID integer,
                type integer,
                unique (itemID, tagID)
            );
            create table collectionItems (
                collectionID integer,
                itemID integer,
                orderIndex integer,
                unique (collectionID, itemID)
            );
            create table itemAnnotations (
                itemID integer,
                parentItemID integer,
                type integer,
                authorName text,
                text text,
                comment text,
                color text,
                pageLabel text,
                sortIndex text,
                position text,
                isExternal integer
            );
            create table itemNotes (
                itemID integer,
                parentItemID integer,
                note text,
                title text
            );
            """
        )
        conn.execute(
            """
            insert into collections
            (collectionName, parentCollectionID, clientDateModified, libraryID, key, version, synced)
            values ('具身智能实验室', null, '2026-06-22 00:00:00', 1, ?, 0, 0)
            """,
            (zd.ROOT_COLLECTION_KEY,),
        )


class ZoteroDeepdivesTests(TestCase):
    def test_summary_note_html_contains_requested_sections(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            page = root / "files" / "paper-deep-dives" / "world-models" / "sample.md"
            page.parent.mkdir(parents=True)
            page.write_text(
                """
---
paper_title: Sample World Model
---

这篇论文的动机是长时机器人学习需要同时处理像素观测、动作和奖励，传统 model-free 方法样本效率太低。

论文的核心贡献可以概括为三点：

1. 提出一个 latent dynamics world model，把像素轨迹压到可规划状态。
2. 使用 action-conditioned rollout 预测 reward，而不是在规划时生成未来图像。
3. 通过在线数据收集和 CEM 规划形成闭环。

## 实验结果

模型在 ImageNet-64 上以 1 forward pass 达到 2.62 FID，接近 512 forward pass teacher 的 2.32 FID。

## 训练细节

训练使用 LAION text-to-image 数据，预计算 50-step teacher outputs；主实验使用 72 A100 GPUs，大约 36 hours。
""",
                encoding="utf-8",
            )
            meta = zd.ArxivMeta(
                arxiv_id="2606.12345",
                title="Sample World Model",
                abstract=(
                    "Long-horizon robot learning remains difficult because agents need compact predictive state. "
                    "We introduce a sample world model that learns latent dynamics from video and action. "
                    "Experiments show improved success rates on simulated manipulation benchmarks."
                ),
                authors=["Ada Lovelace"],
                published="2026-06-01T00:00:00Z",
                updated="2026-06-01T00:00:00Z",
                categories="cs.AI",
                pdf_url="https://arxiv.org/pdf/2606.12345.pdf",
                abs_url="https://arxiv.org/abs/2606.12345",
            )

            note = zd.build_summary_note_html(meta, pages=["world-models/sample.md"], root=root)

        self.assertIn("paper summary", note)
        self.assertNotIn("Codex paper summary", note)
        self.assertIn("主要 idea", note)
        self.assertIn("<ol>", note)
        self.assertEqual(note.count("<li>"), 8)
        self.assertIn("动机", note)
        self.assertIn("核心价值", note)
        self.assertIn("实验效果", note)
        self.assertIn("2.62 FID", note)
        self.assertIn("512 forward pass", note)
        self.assertIn("训练/数据细节", note)
        self.assertIn("LAION text-to-image", note)
        self.assertIn("72 A100 GPUs", note)
        self.assertIn("36 hours", note)
        self.assertIn("https://arxiv.org/abs/2606.12345", note)
        self.assertIn("https://arxiv.org/pdf/2606.12345.pdf", note)
        self.assertIn("file://", note)

    def test_summary_note_prefers_proposal_sentence_for_main_idea(self):
        meta = zd.ArxivMeta(
            arxiv_id="2606.12345",
            title="Sample Planning Agent",
            abstract=(
                "Planning requires dynamics models that are accurate enough for control. "
                "However, learning dynamics models from pixels remains difficult. "
                "We propose a latent planning agent that learns dynamics from images and plans online. "
                "The agent reaches stronger final performance than model-free baselines."
            ),
            authors=["Ada Lovelace"],
            published="2026-06-01T00:00:00Z",
            updated="2026-06-01T00:00:00Z",
            categories="cs.AI",
            pdf_url="https://arxiv.org/pdf/2606.12345.pdf",
            abs_url="https://arxiv.org/abs/2606.12345",
        )

        note = zd.build_summary_note_html(meta, pages=["world-models/sample.md"])

        self.assertIn("We propose a latent planning agent", note)

    def test_summary_note_prefers_three_item_core_contribution_group(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            page = root / "files" / "paper-deep-dives" / "world-models" / "sample.md"
            page.parent.mkdir(parents=True)
            page.write_text(
                """
## 实验结论

Figure 5 的结论更偏系统设计。PlaNet 的效果来自两个闭环同时成立：
图注要点：在线数据收集与 CEM 迭代优化都对任务表现有贡献。

1. planning 让数据收集更接近任务分布；
2. 新数据让 world model 更适合后续 planning。

## 阅读结论

论文的核心贡献可以概括为三点：

1. 提出用 RSSM 从像素学习 action-conditioned latent dynamics；
2. 证明 planner 可以只在 latent space 里评估 reward；
3. 用 latent overshooting 强调多步 rollout 训练。

## 局限与风险

1. 在线规划成本高；
2. planning horizon 有限；
3. 依赖 reward signal。
""",
                encoding="utf-8",
            )
            meta = zd.ArxivMeta(
                arxiv_id="2606.12345",
                title="Sample Planning Agent",
                abstract="We propose a latent planning agent.",
                authors=["Ada Lovelace"],
                published="2026-06-01T00:00:00Z",
                updated="2026-06-01T00:00:00Z",
                categories="cs.AI",
                pdf_url="https://arxiv.org/pdf/2606.12345.pdf",
                abs_url="https://arxiv.org/abs/2606.12345",
            )

            note = zd.build_summary_note_html(meta, pages=["world-models/sample.md"], root=root)

        self.assertIn("提出用 RSSM", note)
        self.assertIn("latent space 里评估 reward", note)
        self.assertIn("latent overshooting", note)
        self.assertNotIn("在线规划成本高", note.split("<ol>", 1)[1].split("</ol>", 1)[0])

    def test_actual_planet_page_uses_reading_conclusion_as_ideas(self):
        root = Path(__file__).resolve().parents[1]
        page = root / "files" / "paper-deep-dives" / "world-models" / "planet.md"
        if not page.exists():
            self.skipTest("PlaNet deep-dive page is not available")
        ideas = zd.extract_numbered_items(page.read_text(encoding="utf-8"))

        self.assertEqual(
            ideas[:3],
            [
                "提出用 RSSM 从像素学习 action-conditioned latent dynamics；",
                "证明 planner 可以只在 latent space 里评估 reward，而不必生成未来图像；",
                "用 latent overshooting 强调 world model 训练必须关注多步 rollout，而不是只优化一步重建。",
            ],
        )

    def test_summary_note_does_not_use_limitations_as_motivation_or_idea(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            page = root / "files" / "paper-deep-dives" / "diffusion" / "sample.md"
            page.parent.mkdir(parents=True)
            page.write_text(
                """
这篇论文的核心思想是把多步 teacher 采样压缩成一步学生模型。

论文把这个目标落成两个损失：

1. Distribution Matching Loss：用 real score 和 fake score 的差异近似 reverse KL 梯度。
2. Regression Loss：预先生成 noise-image pairs，保持 teacher 输出的大尺度结构。

## 局限与风险

1. 仍和 teacher 有质量差距；
2. 训练显存重；
3. 依赖 teacher 能力。
""",
                encoding="utf-8",
            )
            meta = zd.ArxivMeta(
                arxiv_id="2606.12345",
                title="Sample Distillation",
                abstract="We propose a one-step distillation method that matches teacher distributions.",
                authors=["Ada Lovelace"],
                published="2026-06-01T00:00:00Z",
                updated="2026-06-01T00:00:00Z",
                categories="cs.AI",
                pdf_url="https://arxiv.org/pdf/2606.12345.pdf",
                abs_url="https://arxiv.org/abs/2606.12345",
            )

            note = zd.build_summary_note_html(meta, pages=["diffusion/sample.md"], root=root)

        idea_html = note.split("<ol>", 1)[1].split("</ol>", 1)[0]
        motivation = note.split("<p><b>动机：</b>", 1)[1].split("</p>", 1)[0]
        self.assertNotIn("仍和 teacher 有质量差距", idea_html)
        self.assertNotIn("训练显存重", motivation)

    def test_summary_note_compacts_training_table_details(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            page = root / "files" / "paper-deep-dives" / "diffusion" / "sample.md"
            page.parent.mkdir(parents=True)
            page.write_text(
                """
这篇论文的核心思想是做一步扩散蒸馏。

## 训练配置

| Item | Detail |
| --- | --- |
| Dataset | LAION-Aesthetic 6.25+, around 3M images |
| GPUs | 72 A100 GPUs |
| Wall-clock | around 36 hours |
""",
                encoding="utf-8",
            )
            meta = zd.ArxivMeta(
                arxiv_id="2606.12345",
                title="Sample Distillation",
                abstract="We propose a one-step distillation method.",
                authors=["Ada Lovelace"],
                published="2026-06-01T00:00:00Z",
                updated="2026-06-01T00:00:00Z",
                categories="cs.AI",
                pdf_url="https://arxiv.org/pdf/2606.12345.pdf",
                abs_url="https://arxiv.org/abs/2606.12345",
            )

            note = zd.build_summary_note_html(meta, pages=["diffusion/sample.md"], root=root)

        training_html = note.split("<p><b>训练/数据细节：</b></p>", 1)[1].split("</ul>", 1)[0]
        self.assertIn("Dataset：LAION-Aesthetic 6.25+, around 3M images", training_html)
        self.assertIn("GPUs：72 A100 GPUs", training_html)
        self.assertIn("Wall-clock：around 36 hours", training_html)
        self.assertNotIn("| Item | Detail |", training_html)

    def test_actual_dmd_page_does_not_use_limitations_as_motivation_or_idea(self):
        root = Path(__file__).resolve().parents[1]
        page = root / "files" / "paper-deep-dives" / "diffusion" / "dmd.md"
        if not page.exists():
            self.skipTest("DMD deep-dive page is not available")
        meta = zd.ArxivMeta(
            arxiv_id="2311.18828",
            title="One-step Diffusion with Distribution Matching Distillation",
            abstract="We propose distribution matching distillation for one-step diffusion sampling.",
            authors=["Ada Lovelace"],
            published="2023-11-01T00:00:00Z",
            updated="2023-11-01T00:00:00Z",
            categories="cs.CV",
            pdf_url="https://arxiv.org/pdf/2311.18828.pdf",
            abs_url="https://arxiv.org/abs/2311.18828",
        )

        note = zd.build_summary_note_html(meta, pages=["diffusion/dmd.md"], root=root)

        idea_html = note.split("<ol>", 1)[1].split("</ol>", 1)[0]
        motivation = note.split("<p><b>动机：</b>", 1)[1].split("</p>", 1)[0]
        self.assertIn("核心思想", motivation)
        self.assertNotIn("仍和 teacher 有质量差距", idea_html)
        self.assertNotIn("Dimension / DMD", idea_html)
        self.assertNotIn("训练显存重", motivation)

    def test_actual_planet_page_uses_opening_problem_as_motivation(self):
        root = Path(__file__).resolve().parents[1]
        page = root / "files" / "paper-deep-dives" / "world-models" / "planet.md"
        if not page.exists():
            self.skipTest("PlaNet deep-dive page is not available")
        meta = zd.ArxivMeta(
            arxiv_id="1811.04551",
            title="Learning Latent Dynamics for Planning from Pixels",
            abstract="We propose learning latent dynamics for planning from pixels.",
            authors=["Ada Lovelace"],
            published="2018-11-01T00:00:00Z",
            updated="2018-11-01T00:00:00Z",
            categories="cs.LG",
            pdf_url="https://arxiv.org/pdf/1811.04551.pdf",
            abs_url="https://arxiv.org/abs/1811.04551",
        )

        note = zd.build_summary_note_html(meta, pages=["world-models/planet.md"], root=root)

        motivation = note.split("<p><b>动机：</b>", 1)[1].split("</p>", 1)[0]
        self.assertIn("提出的问题很直接", motivation)
        self.assertNotIn("| 维度 |", motivation)

    def test_actual_planet_training_details_skip_comparison_table(self):
        root = Path(__file__).resolve().parents[1]
        page = root / "files" / "paper-deep-dives" / "world-models" / "planet.md"
        if not page.exists():
            self.skipTest("PlaNet deep-dive page is not available")
        meta = zd.ArxivMeta(
            arxiv_id="1811.04551",
            title="Learning Latent Dynamics for Planning from Pixels",
            abstract="We propose learning latent dynamics for planning from pixels.",
            authors=["Ada Lovelace"],
            published="2018-11-01T00:00:00Z",
            updated="2018-11-01T00:00:00Z",
            categories="cs.LG",
            pdf_url="https://arxiv.org/pdf/1811.04551.pdf",
            abs_url="https://arxiv.org/abs/1811.04551",
        )

        note = zd.build_summary_note_html(meta, pages=["world-models/planet.md"], root=root)

        training_html = note.split("<p><b>训练/数据细节：</b></p>", 1)[1].split("</ul>", 1)[0]
        self.assertIn("1,000 episodes", training_html)
        self.assertNotIn("维度 / PlaNet / DreamerV3", training_html)

    def test_summary_note_uses_primary_page_when_arxiv_is_related_elsewhere(self):
        root = Path(__file__).resolve().parents[1]
        planet = root / "files" / "paper-deep-dives" / "world-models" / "planet.md"
        dreamer = root / "files" / "paper-deep-dives" / "world-models" / "dreamer.md"
        if not planet.exists() or not dreamer.exists():
            self.skipTest("World-model deep-dive pages are not available")
        meta = zd.ArxivMeta(
            arxiv_id="1811.04551",
            title="Learning Latent Dynamics for Planning from Pixels",
            abstract="We propose learning latent dynamics for planning from pixels.",
            authors=["Ada Lovelace"],
            published="2018-11-01T00:00:00Z",
            updated="2018-11-01T00:00:00Z",
            categories="cs.LG",
            pdf_url="https://arxiv.org/pdf/1811.04551.pdf",
            abs_url="https://arxiv.org/abs/1811.04551",
        )

        note = zd.build_summary_note_html(
            meta,
            pages=["world-models/dreamer.md", "world-models/planet.md"],
            root=root,
        )

        motivation = note.split("<p><b>动机：</b>", 1)[1].split("</p>", 1)[0]
        self.assertIn("提出的问题很直接", motivation)
        self.assertNotIn("Dreamer 要回答的问题", motivation)
        self.assertIn("file://", note)

    def test_apply_summary_notes_adds_one_idempotent_child_note(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            page = root / "files" / "paper-deep-dives" / "world-models" / "sample.md"
            page.parent.mkdir(parents=True)
            page.write_text("arXiv:2606.12345\n", encoding="utf-8")

            work_dir = root / "tmp" / "zotero-deepdives"
            work_dir.mkdir(parents=True)
            meta = zd.ArxivMeta(
                arxiv_id="2606.12345",
                title="Sample World Model",
                abstract=(
                    "Long-horizon robot learning remains difficult. "
                    "We introduce a sample world model. "
                    "Experiments show improved success rates."
                ),
                authors=["Ada Lovelace"],
                published="2026-06-01T00:00:00Z",
                updated="2026-06-01T00:00:00Z",
                categories="cs.AI",
                pdf_url="https://arxiv.org/pdf/2606.12345.pdf",
                abs_url="https://arxiv.org/abs/2606.12345",
            )
            (work_dir / "arxiv-meta.json").write_text(
                json.dumps({"2606.12345": meta.__dict__}),
                encoding="utf-8",
            )
            db_path = root / "zotero.sqlite"
            _create_minimal_zotero_db(db_path)
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            with conn:
                item_id = zd.create_preprint(conn, meta)

            with patch.object(zd, "ZOTERO_DB", db_path), patch.object(zd, "WORK_DIR", work_dir), patch.object(zd, "META_PATH", work_dir / "arxiv-meta.json"):
                first = zd.apply_summary_notes(root)
                second = zd.apply_summary_notes(root)

            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            notes = conn.execute(
                "select * from itemNotes where parentItemID = ? and note like ?",
                (item_id, f"%{zd.SUMMARY_NOTE_MARKER}%"),
            ).fetchall()
            self.assertEqual(first["created_notes"], 1)
            self.assertEqual(second["created_notes"], 0)
            self.assertEqual(second["updated_notes"], 0)
            self.assertEqual(len(notes), 1)
            self.assertEqual(notes[0]["title"], "paper summary")

    def test_apply_summary_notes_updates_legacy_codex_note_title(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            page = root / "files" / "paper-deep-dives" / "world-models" / "sample.md"
            page.parent.mkdir(parents=True)
            page.write_text("arXiv:2606.12345\n", encoding="utf-8")

            work_dir = root / "tmp" / "zotero-deepdives"
            work_dir.mkdir(parents=True)
            meta = zd.ArxivMeta(
                arxiv_id="2606.12345",
                title="Sample World Model",
                abstract="We propose a sample world model.",
                authors=["Ada Lovelace"],
                published="2026-06-01T00:00:00Z",
                updated="2026-06-01T00:00:00Z",
                categories="cs.AI",
                pdf_url="https://arxiv.org/pdf/2606.12345.pdf",
                abs_url="https://arxiv.org/abs/2606.12345",
            )
            (work_dir / "arxiv-meta.json").write_text(
                json.dumps({"2606.12345": meta.__dict__}),
                encoding="utf-8",
            )
            db_path = root / "zotero.sqlite"
            _create_minimal_zotero_db(db_path)
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            with conn:
                item_id = zd.create_preprint(conn, meta)
                note_item_id = zd.create_summary_note(
                    conn,
                    item_id,
                    '<div><p><b>Codex paper summary</b></p><p>old</p></div>',
                )
                conn.execute(
                    "update itemNotes set title = ? where itemID = ?",
                    ("Codex paper summary", note_item_id),
                )

            with patch.object(zd, "ZOTERO_DB", db_path), patch.object(zd, "WORK_DIR", work_dir), patch.object(zd, "META_PATH", work_dir / "arxiv-meta.json"):
                stats = zd.apply_summary_notes(root)

            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            notes = conn.execute("select * from itemNotes where parentItemID = ?", (item_id,)).fetchall()
            self.assertEqual(stats["created_notes"], 0)
            self.assertEqual(stats["updated_notes"], 1)
            self.assertEqual(len(notes), 1)
            self.assertEqual(notes[0]["title"], "paper summary")
            self.assertIn("paper summary", notes[0]["note"])
            self.assertNotIn("Codex paper summary", notes[0]["note"])

    def test_rand_key_uses_zotero_valid_object_key_characters(self):
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "zotero.sqlite"
            _create_minimal_zotero_db(db_path)
            conn = sqlite3.connect(db_path)

            for _ in range(100):
                key = zd.rand_key(conn)
                self.assertRegex(key, zd.ZOTERO_OBJECT_KEY_RE)
                self.assertNotRegex(key, r"[01O]")

    def test_apply_can_skip_pdf_annotations_while_tagging_and_classifying(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            page = root / "files" / "paper-deep-dives" / "world-models" / "sample.md"
            page.parent.mkdir(parents=True)
            page.write_text("arXiv:2606.12345\n", encoding="utf-8")

            work_dir = root / "tmp" / "zotero-deepdives"
            pdf_dir = work_dir / "pdfs"
            pdf_dir.mkdir(parents=True)
            meta = zd.ArxivMeta(
                arxiv_id="2606.12345",
                title="Sample World Model",
                abstract="We propose a sample world model.",
                authors=["Ada Lovelace"],
                published="2026-06-01T00:00:00Z",
                updated="2026-06-01T00:00:00Z",
                categories="cs.AI",
                pdf_url="https://arxiv.org/pdf/2606.12345.pdf",
                abs_url="https://arxiv.org/abs/2606.12345",
            )
            (work_dir / "arxiv-meta.json").write_text(
                json.dumps({"2606.12345": meta.__dict__}),
                encoding="utf-8",
            )
            (pdf_dir / "2606.12345 - Sample World Model.pdf").write_bytes(b"%PDF-1.4\n")

            db_path = root / "zotero.sqlite"
            storage = root / "storage"
            storage.mkdir()
            _create_minimal_zotero_db(db_path)

            with patch.object(zd, "WORK_DIR", work_dir), patch.object(zd, "PDF_DIR", pdf_dir), patch.object(zd, "META_PATH", work_dir / "arxiv-meta.json"), patch.object(zd, "ZOTERO_DB", db_path), patch.object(zd, "ZOTERO_STORAGE", storage), patch.object(zd, "add_annotations", side_effect=AssertionError("annotations should be skipped")):
                stats = zd.apply(root, skip_annotations=True)

            self.assertEqual(stats["created_items"], 1)
            self.assertEqual(stats["attached_pdfs"], 1)
            self.assertEqual(stats["added_unread_tags"], 1)
            self.assertGreaterEqual(stats["added_collection_memberships"], 1)
            self.assertEqual(stats["added_annotations"], 0)

    def test_repair_invalid_keys_updates_db_and_attachment_storage_folder(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            db_path = root / "zotero.sqlite"
            storage = root / "storage"
            old_folder = storage / "BAD0KEY1"
            old_folder.mkdir(parents=True)
            (old_folder / "paper.pdf").write_bytes(b"%PDF-1.4\n")
            _create_minimal_zotero_db(db_path)
            conn = sqlite3.connect(db_path)
            with conn:
                conn.execute(
                    """
                    insert into items
                    (itemID, itemTypeID, dateAdded, dateModified, clientDateModified, libraryID, key, version, synced)
                    values (10, 3, '2026-06-22 00:00:00', '2026-06-22 00:00:00', '2026-06-22 00:00:00', 1, 'BAD0KEY1', 0, 0)
                    """
                )
                conn.execute(
                    """
                    insert into itemAttachments
                    (itemID, parentItemID, linkMode, contentType, charsetID, path, syncState)
                    values (10, null, 0, 'application/pdf', null, 'storage:paper.pdf', 0)
                    """
                )
                conn.execute(
                    "insert into collections (collectionID, collectionName, parentCollectionID, clientDateModified, libraryID, key, version, synced) values (20, 'Bad', null, '2026-06-22 00:00:00', 1, 'BAD0COL1', 0, 0)"
                )

            with patch.object(zd, "ZOTERO_DB", db_path), patch.object(zd, "ZOTERO_STORAGE", storage), patch.object(zd, "WORK_DIR", root / "work"), patch.object(zd, "PDF_DIR", root / "work" / "pdfs"):
                stats = zd.repair_invalid_keys()

            conn = sqlite3.connect(db_path)
            item_key = conn.execute("select key from items where itemID = 10").fetchone()[0]
            collection_key = conn.execute("select key from collections where collectionID = 20").fetchone()[0]

            self.assertRegex(item_key, zd.ZOTERO_OBJECT_KEY_RE)
            self.assertRegex(collection_key, zd.ZOTERO_OBJECT_KEY_RE)
            self.assertFalse(old_folder.exists())
            self.assertTrue((storage / item_key / "paper.pdf").exists())
            self.assertEqual(stats["repaired_items"], 1)
            self.assertEqual(stats["repaired_collections"], 1)
            self.assertEqual(stats["renamed_storage_dirs"], 1)
