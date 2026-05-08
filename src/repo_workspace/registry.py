"""Task registry for Repo-workspace."""

from __future__ import annotations

from pathlib import Path

from .tasks.spooky_author import MleBenchSpookyAuthorMultifileTask
from .tasks.nomad2018 import MleBenchNomadMultifileTask
from .tasks.aerial_cactus import MleBenchAerialCactusMultifileTask
from .tasks.spaceship_titanic import MleBenchSpaceshipTitanicMultifileTask
from .tasks.random_acts_of_pizza import MleBenchRandomActsOfPizzaMultifileTask
from .tasks.learning_agency_essay_scoring_2 import MleBenchLearningAgencyEssayScoring2MultifileTask
from .tasks.google_quest import MleBenchGoogleQuestMultifileTask
from .tasks.text_normalization_english import MleBenchTextNormalizationEnglishMultifileTask
from .tasks.dog_breed_identification import MleBenchDogBreedIdentificationMultifileTask
from .tasks.plant_pathology_2020 import MleBenchPlantPathology2020MultifileTask
from .tasks.petfinder_pawpularity import MleBenchPetfinderPawpularityMultifileTask
from .tasks.leaf_classification import MleBenchLeafClassificationMultifileTask
from .tasks.text_normalization_russian import MleBenchTextNormalizationRussianMultifileTask
from .tasks.denoising_dirty_documents import MleBenchDenoisingDirtyDocumentsMultifileTask
from .tasks.house_prices import MleBenchHousePricesMultifileTask
from .tasks.titanic import MleBenchTitanicMultifileTask
from .tasks.santander_value import MleBenchSantanderValueMultifileTask
from .tasks.mercedes_benz import MleBenchMercedesBenzMultifileTask
from .tasks.restaurant_revenue import MleBenchRestaurantRevenueMultifileTask
from .tasks.icr_age_related_conditions import MleBenchIcrAgeRelatedConditionsMultifileTask
from .tasks.forest_cover_type import MleBenchForestCoverTypeMultifileTask
from .tasks.nlp_getting_started import MleBenchNlpGettingStartedMultifileTask
from .tasks.crowdflower_search_relevance import MleBenchCrowdflowerSearchRelevanceMultifileTask
from .tasks.commonlit_readability import MleBenchCommonLitReadabilityMultifileTask
from .tasks.feedback_english_language_learning import MleBenchFeedbackEnglishLanguageLearningMultifileTask
from .tasks.feedback_effectiveness import MleBenchFeedbackEffectivenessMultifileTask
from .tasks.transfer_learning_stack_exchange_tags import MleBenchTransferLearningStackExchangeTagsMultifileTask
from .tasks.facial_keypoints_detection import MleBenchFacialKeypointsDetectionMultifileTask
from .tasks.data_science_bowl_2018 import MleBenchDataScienceBowl2018MultifileTask
from .tasks.kuzushiji_recognition import MleBenchKuzushijiRecognitionMultifileTask
from .tasks.kvasir_seg import MleBenchKvasirSegMultifileTask
from .tasks.cofw_face_landmarks import MleBenchCofwFaceLandmarksMultifileTask
from .tasks.cmu_hand_keypoints import MleBenchCmuHandKeypointsMultifileTask
from .tasks.tgs_salt_identification import MleBenchTgsSaltIdentificationMultifileTask
from .tasks.uw_madison_gi_tract_image_segmentation import MleBenchUwMadisonGiTractImageSegmentationMultifileTask


def _resolve_path(repo_root: Path, raw_path: str | Path | None) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path


def build_task_registry(repo_root: Path, config: dict | None = None) -> dict[str, object]:
    template_root = repo_root / "templates" / "repo_workspace_bounded_task_workspace"
    registry: dict[str, object] = {}

    if not config:
        return registry

    mle_bench_data_root = config.get("mle_bench_data_root")
    mle_bench_fallback_data_root = config.get("mle_bench_fallback_data_root")
    if not mle_bench_data_root:
        return registry

    # Prepared tasks consume local templates plus packaged benchmark data.
    data_path = _resolve_path(repo_root, mle_bench_data_root)
    fallback_data_path = _resolve_path(repo_root, mle_bench_fallback_data_root)

    if data_path is None:
        return registry
    if not data_path.exists() and not (fallback_data_path is not None and fallback_data_path.exists()):
        return registry

    registry["mlebench_spooky_author_multifile"] = MleBenchSpookyAuthorMultifileTask(
        name="mlebench_spooky_author_multifile",
        template_dir=template_root / "mlebench_spooky_author_multifile",
        mle_bench_data_root=data_path,
        mle_bench_fallback_data_root=fallback_data_path,
        competition_id="spooky-author-identification",
    )
    registry["mlebench_nomad2018_multifile"] = MleBenchNomadMultifileTask(
        name="mlebench_nomad2018_multifile",
        template_dir=template_root / "mlebench_nomad2018_multifile",
        mle_bench_data_root=data_path,
        mle_bench_fallback_data_root=fallback_data_path,
        competition_id="nomad2018-predict-transparent-conductors",
    )
    registry["mlebench_aerial_cactus_multifile"] = MleBenchAerialCactusMultifileTask(
        name="mlebench_aerial_cactus_multifile",
        template_dir=template_root / "mlebench_aerial_cactus_multifile",
        mle_bench_data_root=data_path,
        mle_bench_fallback_data_root=fallback_data_path,
        competition_id="aerial-cactus-identification",
    )
    registry["mlebench_spaceship_titanic_multifile"] = MleBenchSpaceshipTitanicMultifileTask(
        name="mlebench_spaceship_titanic_multifile",
        template_dir=template_root / "mlebench_spaceship_titanic_multifile",
        mle_bench_data_root=data_path,
        mle_bench_fallback_data_root=fallback_data_path,
        competition_id="spaceship-titanic",
    )
    registry["mlebench_random_acts_of_pizza_multifile"] = MleBenchRandomActsOfPizzaMultifileTask(
        name="mlebench_random_acts_of_pizza_multifile",
        template_dir=template_root / "mlebench_random_acts_of_pizza_multifile",
        mle_bench_data_root=data_path,
        mle_bench_fallback_data_root=fallback_data_path,
        competition_id="random-acts-of-pizza",
    )
    registry["mlebench_learning_agency_essay_scoring_2_multifile"] = MleBenchLearningAgencyEssayScoring2MultifileTask(
        name="mlebench_learning_agency_essay_scoring_2_multifile",
        template_dir=template_root / "mlebench_learning_agency_essay_scoring_2_multifile",
        mle_bench_data_root=data_path,
        mle_bench_fallback_data_root=fallback_data_path,
        competition_id="learning-agency-lab-automated-essay-scoring-2",
    )
    registry["mlebench_google_quest_multifile"] = MleBenchGoogleQuestMultifileTask(
        name="mlebench_google_quest_multifile",
        template_dir=template_root / "mlebench_google_quest_multifile",
        mle_bench_data_root=data_path,
        mle_bench_fallback_data_root=fallback_data_path,
        competition_id="google-quest-challenge",
    )
    registry["mlebench_text_normalization_english_multifile"] = MleBenchTextNormalizationEnglishMultifileTask(
        name="mlebench_text_normalization_english_multifile",
        template_dir=template_root / "mlebench_text_normalization_english_multifile",
        mle_bench_data_root=data_path,
        mle_bench_fallback_data_root=fallback_data_path,
        competition_id="text-normalization-challenge-english-language",
    )
    registry["mlebench_dog_breed_identification_multifile"] = MleBenchDogBreedIdentificationMultifileTask(
        name="mlebench_dog_breed_identification_multifile",
        template_dir=template_root / "mlebench_dog_breed_identification_multifile",
        mle_bench_data_root=data_path,
        mle_bench_fallback_data_root=fallback_data_path,
        competition_id="dog-breed-identification",
    )
    registry["mlebench_plant_pathology_2020_multifile"] = MleBenchPlantPathology2020MultifileTask(
        name="mlebench_plant_pathology_2020_multifile",
        template_dir=template_root / "mlebench_plant_pathology_2020_multifile",
        mle_bench_data_root=data_path,
        mle_bench_fallback_data_root=fallback_data_path,
        competition_id="plant-pathology-2020-fgvc7",
    )
    registry["mlebench_petfinder_pawpularity_multifile"] = MleBenchPetfinderPawpularityMultifileTask(
        name="mlebench_petfinder_pawpularity_multifile",
        template_dir=template_root / "mlebench_petfinder_pawpularity_multifile",
        mle_bench_data_root=data_path,
        mle_bench_fallback_data_root=fallback_data_path,
        competition_id="petfinder-pawpularity-score",
    )
    registry["mlebench_leaf_classification_multifile"] = MleBenchLeafClassificationMultifileTask(
        name="mlebench_leaf_classification_multifile",
        template_dir=template_root / "mlebench_leaf_classification_multifile",
        mle_bench_data_root=data_path,
        mle_bench_fallback_data_root=fallback_data_path,
        competition_id="leaf-classification",
    )
    registry["mlebench_text_normalization_russian_multifile"] = MleBenchTextNormalizationRussianMultifileTask(
        name="mlebench_text_normalization_russian_multifile",
        template_dir=template_root / "mlebench_text_normalization_russian_multifile",
        mle_bench_data_root=data_path,
        mle_bench_fallback_data_root=fallback_data_path,
        competition_id="text-normalization-challenge-russian-language",
    )
    registry["mlebench_denoising_dirty_documents_multifile"] = MleBenchDenoisingDirtyDocumentsMultifileTask(
        name="mlebench_denoising_dirty_documents_multifile",
        template_dir=template_root / "mlebench_denoising_dirty_documents_multifile",
        mle_bench_data_root=data_path,
        mle_bench_fallback_data_root=fallback_data_path,
        competition_id="denoising-dirty-documents",
    )
    registry["mlebench_house_prices_multifile"] = MleBenchHousePricesMultifileTask(
        name="mlebench_house_prices_multifile",
        template_dir=template_root / "mlebench_house_prices_multifile",
        mle_bench_data_root=data_path,
        mle_bench_fallback_data_root=fallback_data_path,
        competition_id="house-prices-advanced-regression-techniques",
    )
    registry["mlebench_titanic_multifile"] = MleBenchTitanicMultifileTask(
        name="mlebench_titanic_multifile",
        template_dir=template_root / "mlebench_titanic_multifile",
        mle_bench_data_root=data_path,
        mle_bench_fallback_data_root=fallback_data_path,
        competition_id="titanic",
    )
    registry["mlebench_santander_value_multifile"] = MleBenchSantanderValueMultifileTask(
        name="mlebench_santander_value_multifile",
        template_dir=template_root / "mlebench_santander_value_multifile",
        mle_bench_data_root=data_path,
        mle_bench_fallback_data_root=fallback_data_path,
        competition_id="santander-value-prediction-challenge",
    )
    registry["mlebench_mercedes_benz_multifile"] = MleBenchMercedesBenzMultifileTask(
        name="mlebench_mercedes_benz_multifile",
        template_dir=template_root / "mlebench_mercedes_benz_multifile",
        mle_bench_data_root=data_path,
        mle_bench_fallback_data_root=fallback_data_path,
        competition_id="mercedes-benz-greener-manufacturing",
    )
    registry["mlebench_restaurant_revenue_multifile"] = MleBenchRestaurantRevenueMultifileTask(
        name="mlebench_restaurant_revenue_multifile",
        template_dir=template_root / "mlebench_restaurant_revenue_multifile",
        mle_bench_data_root=data_path,
        mle_bench_fallback_data_root=fallback_data_path,
        competition_id="restaurant-revenue-prediction",
    )
    registry["mlebench_icr_age_related_conditions_multifile"] = MleBenchIcrAgeRelatedConditionsMultifileTask(
        name="mlebench_icr_age_related_conditions_multifile",
        template_dir=template_root / "mlebench_icr_age_related_conditions_multifile",
        mle_bench_data_root=data_path,
        mle_bench_fallback_data_root=fallback_data_path,
        competition_id="icr-identify-age-related-conditions",
    )
    registry["mlebench_forest_cover_type_multifile"] = MleBenchForestCoverTypeMultifileTask(
        name="mlebench_forest_cover_type_multifile",
        template_dir=template_root / "mlebench_forest_cover_type_multifile",
        mle_bench_data_root=data_path,
        mle_bench_fallback_data_root=fallback_data_path,
        competition_id="forest-cover-type-kernels-only",
    )
    registry["mlebench_nlp_getting_started_multifile"] = MleBenchNlpGettingStartedMultifileTask(
        name="mlebench_nlp_getting_started_multifile",
        template_dir=template_root / "mlebench_nlp_getting_started_multifile",
        mle_bench_data_root=data_path,
        mle_bench_fallback_data_root=fallback_data_path,
        competition_id="nlp-getting-started",
    )
    registry["mlebench_crowdflower_search_relevance_multifile"] = MleBenchCrowdflowerSearchRelevanceMultifileTask(
        name="mlebench_crowdflower_search_relevance_multifile",
        template_dir=template_root / "mlebench_crowdflower_search_relevance_multifile",
        mle_bench_data_root=data_path,
        mle_bench_fallback_data_root=fallback_data_path,
        competition_id="crowdflower-search-relevance",
    )
    registry["mlebench_commonlit_readability_multifile"] = MleBenchCommonLitReadabilityMultifileTask(
        name="mlebench_commonlit_readability_multifile",
        template_dir=template_root / "mlebench_commonlit_readability_multifile",
        mle_bench_data_root=data_path,
        mle_bench_fallback_data_root=fallback_data_path,
        competition_id="commonlitreadabilityprize",
    )
    registry["mlebench_feedback_english_language_learning_multifile"] = MleBenchFeedbackEnglishLanguageLearningMultifileTask(
        name="mlebench_feedback_english_language_learning_multifile",
        template_dir=template_root / "mlebench_feedback_english_language_learning_multifile",
        mle_bench_data_root=data_path,
        mle_bench_fallback_data_root=fallback_data_path,
        competition_id="feedback-prize-english-language-learning",
    )
    registry["mlebench_feedback_effectiveness_multifile"] = MleBenchFeedbackEffectivenessMultifileTask(
        name="mlebench_feedback_effectiveness_multifile",
        template_dir=template_root / "mlebench_feedback_effectiveness_multifile",
        mle_bench_data_root=data_path,
        mle_bench_fallback_data_root=fallback_data_path,
        competition_id="feedback-prize-effectiveness",
    )
    registry["mlebench_transfer_learning_stack_exchange_tags_multifile"] = MleBenchTransferLearningStackExchangeTagsMultifileTask(
        name="mlebench_transfer_learning_stack_exchange_tags_multifile",
        template_dir=template_root / "mlebench_transfer_learning_stack_exchange_tags_multifile",
        mle_bench_data_root=data_path,
        mle_bench_fallback_data_root=fallback_data_path,
        competition_id="transfer-learning-on-stack-exchange-tags",
    )
    registry["mlebench_facial_keypoints_detection_multifile"] = MleBenchFacialKeypointsDetectionMultifileTask(
        name="mlebench_facial_keypoints_detection_multifile",
        template_dir=template_root / "mlebench_facial_keypoints_detection_multifile",
        mle_bench_data_root=data_path,
        mle_bench_fallback_data_root=fallback_data_path,
        competition_id="facial-keypoints-detection",
    )
    registry["mlebench_data_science_bowl_2018_multifile"] = MleBenchDataScienceBowl2018MultifileTask(
        name="mlebench_data_science_bowl_2018_multifile",
        template_dir=template_root / "mlebench_data_science_bowl_2018_multifile",
        mle_bench_data_root=data_path,
        mle_bench_fallback_data_root=fallback_data_path,
        competition_id="data-science-bowl-2018",
    )
    registry["mlebench_kuzushiji_recognition_multifile"] = MleBenchKuzushijiRecognitionMultifileTask(
        name="mlebench_kuzushiji_recognition_multifile",
        template_dir=template_root / "mlebench_kuzushiji_recognition_multifile",
        mle_bench_data_root=data_path,
        mle_bench_fallback_data_root=fallback_data_path,
        competition_id="kuzushiji-recognition",
    )
    registry["mlebench_kvasir_seg_multifile"] = MleBenchKvasirSegMultifileTask(
        name="mlebench_kvasir_seg_multifile",
        template_dir=template_root / "mlebench_kvasir_seg_multifile",
        mle_bench_data_root=data_path,
        mle_bench_fallback_data_root=fallback_data_path,
        competition_id="kvasir-seg",
    )
    registry["mlebench_cofw_face_landmarks_multifile"] = MleBenchCofwFaceLandmarksMultifileTask(
        name="mlebench_cofw_face_landmarks_multifile",
        template_dir=template_root / "mlebench_cofw_face_landmarks_multifile",
        mle_bench_data_root=data_path,
        mle_bench_fallback_data_root=fallback_data_path,
        competition_id="cofw-face-landmarks",
    )
    registry["mlebench_cmu_hand_keypoints_multifile"] = MleBenchCmuHandKeypointsMultifileTask(
        name="mlebench_cmu_hand_keypoints_multifile",
        template_dir=template_root / "mlebench_cmu_hand_keypoints_multifile",
        mle_bench_data_root=data_path,
        mle_bench_fallback_data_root=fallback_data_path,
        competition_id="cmu-hand-keypoints",
    )
    registry["mlebench_tgs_salt_identification_multifile"] = MleBenchTgsSaltIdentificationMultifileTask(
        name="mlebench_tgs_salt_identification_multifile",
        template_dir=template_root / "mlebench_tgs_salt_identification_multifile",
        mle_bench_data_root=data_path,
        mle_bench_fallback_data_root=fallback_data_path,
        competition_id="tgs-salt-identification-challenge",
    )
    registry["mlebench_uw_madison_gi_tract_image_segmentation_multifile"] = MleBenchUwMadisonGiTractImageSegmentationMultifileTask(
        name="mlebench_uw_madison_gi_tract_image_segmentation_multifile",
        template_dir=template_root / "mlebench_uw_madison_gi_tract_image_segmentation_multifile",
        mle_bench_data_root=data_path,
        mle_bench_fallback_data_root=fallback_data_path,
        competition_id="uw-madison-gi-tract-image-segmentation",
    )
    return registry


def get_task(repo_root: Path, task_name: str, config: dict | None = None) -> object:
    registry = build_task_registry(repo_root, config=config)
    if task_name not in registry:
        raise KeyError(f"Unknown Repo-workspace task: {task_name}")
    return registry[task_name]
