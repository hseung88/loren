from .glue_transformer import GLUETransformer
from .glue_transformer_fo import GLUETransformerFO
from .glue_transformer_mezo import GLUETransformerMeZo
from .glue_transformer_svrg import GLUETransformerMeZoSVRG
from .glue_transformer_adam import GLUETransformerMeZoAdam
from .glue_transformer_lozo import GLUETransformerLoZo
from .glue_transformer_hizoo import GLUETransformerHiZOO
from .glue_transformer_loren import GLUETransformerLOREN
from .opt import OPTDecoder, OPTForCausalLM, OPTModel
from .roberta import RobertaConfig, RobertaLMHead, RobertaModel, RobertaPreTrainedModel
from .prompt_finetune import (RobertaModelForPromptFinetuning, OPTModelForPromptFinetuning,
                              LlamaModelForPromptFinetuning)


__all__ = ['GLUETransformer', 'GLUETransformerMeZo',
           'GLUETransformerMeZoSVRG', 'GLUETransformerMeZoAdam',
           'GLUETransformerHiZOO', 'GLUETransformerLOREN',
           'RobertaPreTrainedModel', 'RobertaModel', 'RobertaLMHead',
           'RobertaModelForPromptFinetuning', 'RobertaConfig', 'OPTModelForPromptFinetuning',
           'OPTModel', 'OPTForCausalLM', 'OPTDecoder',  'LlamaModelForPromptFinetuning']


model_class = {
    'FOSGD': GLUETransformerFO,
    'FOAdam': GLUETransformerFO,
    'ZO': GLUETransformerMeZo,
    'ZOSVRG': GLUETransformerMeZoSVRG,
    'ZOAdam': GLUETransformerMeZoAdam,
    'LOZO': GLUETransformerLoZo,
    'HiZOO': GLUETransformerHiZOO,
    'LOREN': GLUETransformerLOREN,
}


def get_model(args, dm):
    glue_transformer = model_class[args.algorithm]
    transformer = glue_transformer(
        model_name_or_path=args.model_name,
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        task_name=dm.task_name,
        learning_rate=args.lr_mezosvrg_mb if 'SVRG' in args.algorithm else args.lr,
        learning_rate_aux=args.lr if 'SVRG' in args.algorithm else None,
        lr_anneal=args.anneal,
        full_parameter=args.full_parameter,
        batchsize_limit=args.batchsize_limit,
        zero_order_eps=args.perturbation_scale,
        soft_prompt=args.soft_prompt,
        use_SGD=True if 'SGD' in args.algorithm else False,
        logger_type=args.logging,
        hf_token=args.hf_token,
        model_init_seed=args.init_seed,
        q=args.q,
        step_interval=args.step_interval,
        rank=args.rank,
        weight_decay=args.weight_decay,
        eps=args.eps,
        damping=args.damping,
        lr_cov=args.lr_cov,
        num_queries=args.num_queries,
        clipping=args.clipping,
        hessian_smooth_type=args.hessian_smooth_type,
        num_epochs=args.epochs,
    )
    return transformer
