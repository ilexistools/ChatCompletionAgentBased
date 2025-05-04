def improve_gpt_prompt_by_ai(
    agent: GPTAgent,
    training_data: List[dict],
    trainer_model=default_model,
    **kwargs
) -> str:
    """
    Automatically improve a GPT agent’s prompt via AI-driven training, evaluation, and synthesis.

    Parameters:
        agent (GPTAgent): The initial agent whose prompt and configuration will be optimized.
        training_data (List[dict]): A collection of input instances and expected outputs for training.
        trainer_model (str): Identifier of the GPT model to use for training and evaluation (default is default_model).
        **kwargs:
            verbose (bool): If True, prints colorized progress and debug messages to the console.

    Returns:
        str: The final synthesized prompt instructions for the improved agent.
    """

    # Definições de cores ANSI para realce
    COLORS = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
    }
    def color_text(text: str, color: str = 'reset') -> str:
        return f"{COLORS.get(color, COLORS['reset'])}{text}{COLORS['reset']}"

    verbose = kwargs.get('verbose', False)
    log = print if verbose else lambda *args, **kwargs: None

    evaluator = build('_trainer_evaluator')
    evaluator.gpt_model = trainer_model
    evaluator.max_tokens = default_max_tokens

    factory = GPTFactory()
    def make_trainer(
        name: str,
        max_tokens: int = default_max_tokens,
        temperature: float = None,
        json_format: str = None,
    ):
        t = factory.build(name)
        t.gpt_model = trainer_model
        t.max_tokens = max_tokens
        if temperature is not None:
            t.temperature = temperature
        if json_format is not None:
            t.json_format = json_format
        return t

    # Fase 1: Execução e avaliação
    log(color_text('--- Iniciando avaliação inicial ---', 'blue'))
    results = []
    total = len(training_data)
    for idx, instance in enumerate(training_data, start=1):
        log(color_text(f"[{idx}/{total}] Executando agente...", 'yellow'))
        try:
            output = agent.run(inputs=instance)
            eval_resp = evaluator.run(inputs={
                'prompt': agent.get_description(),
                'input': instance,
                'output': output,
            })
            results.append((instance, output, eval_resp))
            log(color_text('✔ Output obtido com sucesso', 'green'))
        except Exception as e:
            log(color_text(f"✖ Erro na instância {idx}: {e}", 'red'))

    log(color_text('--- Avaliação inicial concluída ---', 'blue'))

    # Fase 2: Geração de instruções por instância
    log(color_text('--- Gerando instruções de fine-tuning ---', 'blue'))
    trainer = make_trainer('_trainer')
    instructions = []
    for instance, output, eval_resp in results:
        instr = trainer.run(inputs={
            'agent_description': agent.get_description(),
            'instance_data': instance,
            'result': output,
            'human_evaluation': eval_resp,
        })
        instructions.append(instr)
    log(color_text('✔ Instruções individuais geradas', 'green'))

    # Fase 3: Validação binária das instruções
    log(color_text('--- Validando instruções ---', 'blue'))
    validator = make_trainer(
        '_trainer_binary_validator',
        max_tokens=200,
        temperature=0.2,
        json_format='{result: int}'
    )
    human_scores = [
        validator.run(inputs={'evaluation': instr}).get('result', 0)
        for instr in instructions
    ]
    log(color_text('✔ Validação completa', 'green'))

    # Fase 4: Síntese de instruções finais
    log(color_text('--- Sintetizando instruções finais ---', 'blue'))
    synthesizer = make_trainer('_trainer_synthesizer')
    final_instructions = synthesizer.run(inputs={
        'agent_description': agent.get_description(),
        'training_instructions': '\n'.join(instructions),
    })
    log(color_text('✔ Instruções finais sintetizadas', 'green'))

    # Fase 5: Extração de novo config JSON
    log(color_text('--- Extraindo configuração do agente melhorado ---', 'blue'))
    extractor = make_trainer(
        '_json_extractor',
        json_format='{role: str, goal: str, backstory: str, knowledge: str}'
    )
    config = extractor.run(inputs={'text': final_instructions})

    new_agent = create_agent(
        config['role'],
        config['goal'],
        config['backstory'],
        config['knowledge'],
    )
    log(color_text('✔ Novo agente criado', 'green'))

    # Fase 6: Teste do novo agente e cálculo de precisão
    log(color_text('--- Testando agente melhorado ---', 'blue'))
    test_scores = []
    for idx, instance in enumerate(training_data, start=1):
        try:
            output = new_agent.run(inputs=instance)
            eval_resp = evaluator.run(inputs={
                'prompt': agent.get_description(),
                'input': instance,
                'output': output,
            })
            valid = validator.run(inputs={'evaluation': eval_resp}).get('result', 0)
            test_scores.append((instance, output, valid))
            log(color_text(f"[{idx}] Resultado validado: {valid}", 'yellow'))
        except Exception as e:
            log(color_text(f"✖ Erro no teste da instância {idx}: {e}", 'red'))

    tp = sum(1 for gs, ts in zip(human_scores, test_scores) if gs == 1 and ts[2] == 1)
    fp = sum(1 for gs, ts in zip(human_scores, test_scores) if gs == 0 and ts[2] == 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    print(color_text(f"Precision final: {precision:.2f}", 'bold'))

    # Fase 7: Salvando resultados
    util.write_all_text(
        'improved_results.txt',
        '\n\n'.join(
            f"{i+1}. {inst}\n{out}" for i, (inst, out, _) in enumerate(test_scores)
        )
    )
    util.write_all_text('config/improved_agent.yaml', final_instructions)

    return final_instructions

def improve_gpt_prompt_by_human(
    agent: GPTAgent,
    training_data: List[dict],
    trainer_model=default_model,
    **kwargs
) -> str:
    """
    Interactively improve a GPT agent’s prompt using human-in-the-loop evaluation and AI-driven synthesis.

    Parameters:
        agent (GPTAgent): The initial agent whose prompt and configuration will be optimized.
        training_data (List[dict]): A collection of input instances and their outputs for human review.
        trainer_model (str): Identifier of the GPT model to use for instruction generation and validation (default is default_model).
        **kwargs:
            verbose (bool): If True, prints colorized progress and debug messages to the console.

    Returns:
        str: The final synthesized prompt instructions for the improved agent.
    """

    # Definições de cores ANSI para realce
    COLORS = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
    }
    def color_text(text: str, color: str = 'reset') -> str:
        return f"{COLORS.get(color, COLORS['reset'])}{text}{COLORS['reset']}"

    verbose = kwargs.get('verbose', False)
    log = print if verbose else lambda *args, **kwargs: None

    # Avaliador automático (usado apenas na fase de teste)
    evaluator = build('_trainer_evaluator')
    evaluator.gpt_model = trainer_model
    evaluator.max_tokens = default_max_tokens

    factory = GPTFactory()
    def make_trainer(
        name: str,
        max_tokens: int = default_max_tokens,
        temperature: float = None,
        json_format: str = None,
    ):
        t = factory.build(name)
        t.gpt_model = trainer_model
        t.max_tokens = max_tokens
        if temperature is not None:
            t.temperature = temperature
        if json_format is not None:
            t.json_format = json_format
        return t

    # Fase 1: Execução e avaliação HUMANA via input() com comentário
    log(color_text('--- Iniciando avaliação humana via input() ---', 'blue'))
    results = []
    total = len(training_data)
    for idx, instance in enumerate(training_data, start=1):
        log(color_text(f"[{idx}/{total}] Executando agente...", 'yellow'))
        try:
            output = agent.run(inputs=instance)
            # Exibir instância e saída para avaliação manual
            print(color_text(f"Instância: {instance}", 'yellow'))
            print(color_text(f"Saída: {output}", 'yellow'))
            # Solicitar comentário livre
            comment = input(color_text("Comentário sobre a saída: ", 'bold'))
            results.append((instance, output, comment))
            log(color_text(f"Comentário recebido: {comment}", 'green'))
        except Exception as e:
            log(color_text(f"✖ Erro na instância {idx}: {e}", 'red'))

    log(color_text('--- Avaliação humana concluída ---', 'blue'))

    # Fase 2: Geração de instruções por instância
    log(color_text('--- Gerando instruções de fine-tuning ---', 'blue'))
    trainer = make_trainer('_trainer')
    instructions = []
    for instance, output, human_comment in results:
        instr = trainer.run(inputs={
            'agent_description': agent.get_description(),
            'instance_data': instance,
            'result': output,
            'human_evaluation': human_comment,
        })
        instructions.append(instr)
    log(color_text('✔ Instruções individuais geradas', 'green'))

    # Fase 3: Validação binária das instruções
    log(color_text('--- Validando instruções ---', 'blue'))
    validator = make_trainer(
        '_trainer_binary_validator',
        max_tokens=200,
        temperature=0.2,
        json_format='{result: int}'
    )
    human_scores = [
        validator.run(inputs={'evaluation': instr}).get('result', 0)
        for instr in instructions
    ]
    log(color_text('✔ Validação completa', 'green'))

    # Fase 4: Síntese de instruções finais
    log(color_text('--- Sintetizando instruções finais ---', 'blue'))
    synthesizer = make_trainer('_trainer_synthesizer')
    final_instructions = synthesizer.run(inputs={
        'agent_description': agent.get_description(),
        'training_instructions': '\n'.join(instructions),
    })
    log(color_text('✔ Instruções finais sintetizadas', 'green'))

    # Fase 5: Extração de novo config JSON
    log(color_text('--- Extraindo configuração do agente melhorado ---', 'blue'))
    extractor = make_trainer(
        '_json_extractor',
        json_format='{role: str, goal: str, backstory: str, knowledge: str}'
    )
    config = extractor.run(inputs={'text': final_instructions})

    new_agent = create_agent(
        config['role'],
        config['goal'],
        config['backstory'],
        config['knowledge'],
    )
    log(color_text('✔ Novo agente criado', 'green'))

    # Fase 6: Teste do novo agente e cálculo de precisão
    log(color_text('--- Testando agente melhorado ---', 'blue'))
    test_scores = []
    for idx, instance in enumerate(training_data, start=1):
        try:
            output = new_agent.run(inputs=instance)
            eval_resp = evaluator.run(inputs={
                'prompt': agent.get_description(),
                'input': instance,
                'output': output,
            })
            valid = validator.run(inputs={'evaluation': eval_resp}).get('result', 0)
            test_scores.append((instance, output, valid))
            log(color_text(f"[{idx}] Resultado validado: {valid}", 'yellow'))
        except Exception as e:
            log(color_text(f"✖ Erro no teste da instância {idx}: {e}", 'red'))

    tp = sum(1 for gs, ts in zip(human_scores, test_scores) if gs == 1 and ts[2] == 1)
    fp = sum(1 for gs, ts in zip(human_scores, test_scores) if gs == 0 and ts[2] == 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    print(color_text(f"Precision final: {precision:.2f}", 'bold'))

    # Fase 7: Salvando resultados
    util.write_all_text(
        'improved_results.txt',
        '\n\n'.join(
            f"{i+1}. {inst}\n{out}" for i, (inst, out, _) in enumerate(test_scores)
        )
    )
    util.write_all_text('config/improved_agent.yaml', final_instructions)

    return final_instructions

def improve_gpt_prompt_by_data(
    agent: GPTAgent,
    training_data: List[dict],
    expected_outputs: List[str],
    trainer_model=default_model,
    **kwargs
) -> str:
    """
    Automatically improve a GPT agent’s prompt using data-driven training, evaluation, and synthesis.

    Parameters:
        agent (GPTAgent): The initial agent whose prompt and configuration will be optimized.
        training_data (List[dict]): A collection of input instances to feed the agent.
        expected_outputs (List[str]): The corresponding expected outputs for each training instance.
        trainer_model (str): Identifier of the GPT model to use for training and evaluation (default is default_model).
        **kwargs:
            verbose (bool): If True, prints colorized progress and debug messages to the console.

    Returns:
        str: The final synthesized prompt instructions for the improved agent.
    """

    # Definições de cores ANSI para realce
    COLORS = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
    }
    def color_text(text: str, color: str = 'reset') -> str:
        return f"{COLORS.get(color, COLORS['reset'])}{text}{COLORS['reset']}"

    verbose = kwargs.get('verbose', False)
    log = print if verbose else lambda *args, **kwargs: None

    # Avaliador automático (usado apenas na fase de teste)
    evaluator = build('_trainer_evaluator')
    evaluator.gpt_model = trainer_model
    evaluator.max_tokens = default_max_tokens

    factory = GPTFactory()
    def make_trainer(
        name: str,
        max_tokens: int = default_max_tokens,
        temperature: float = None,
        json_format: str = None,
    ):
        t = factory.build(name)
        t.gpt_model = trainer_model
        t.max_tokens = max_tokens
        if temperature is not None:
            t.temperature = temperature
        if json_format is not None:
            t.json_format = json_format
        return t

    # Fase 1: Execução e coleta de pares (output, expected)
    log(color_text('--- Iniciando avaliação por dados esperados ---', 'blue'))
    results = []
    total = len(training_data)
    for idx, (instance, expected) in enumerate(zip(training_data, expected_outputs), start=1):
        log(color_text(f"[{idx}/{total}] Executando agente...", 'yellow'))
        try:
            output = agent.run(inputs=instance)
            log(color_text(f"Output: {output}", 'yellow'))
            log(color_text(f"Expected: {expected}", 'yellow'))
            results.append((instance, output, expected))
        except Exception as e:
            log(color_text(f"✖ Erro na instância {idx}: {e}", 'red'))

    log(color_text('--- Coleta de resultados concluída ---', 'blue'))

    # Fase 2: Geração de instruções por instância
    log(color_text('--- Gerando instruções de fine-tuning ---', 'blue'))
    trainer = make_trainer('_trainer')
    instructions = []
    for instance, output, expected in results:
        instr = trainer.run(inputs={
            'agent_description': agent.get_description(),
            'instance_data': instance,
            'result': output,
            'human_evaluation': expected,
        })
        instructions.append(instr)
    log(color_text('✔ Instruções individuais geradas', 'green'))

    # Fase 3: Validação binária das instruções
    log(color_text('--- Validando instruções ---', 'blue'))
    validator = make_trainer(
        '_trainer_binary_validator',
        max_tokens=200,
        temperature=0.2,
        json_format='{result: int}'
    )
    instr_scores = [
        validator.run(inputs={'evaluation': instr}).get('result', 0)
        for instr in instructions
    ]
    log(color_text('✔ Validação completa', 'green'))

    # Fase 4: Síntese de instruções finais
    log(color_text('--- Sintetizando instruções finais ---', 'blue'))
    synthesizer = make_trainer('_trainer_synthesizer')
    final_instructions = synthesizer.run(inputs={
        'agent_description': agent.get_description(),
        'training_instructions': '\n'.join(instructions),
    })
    log(color_text('✔ Instruções finais sintetizadas', 'green'))

    # Fase 5: Extração de novo config JSON
    log(color_text('--- Extraindo configuração do agente melhorado ---', 'blue'))
    extractor = make_trainer(
        '_json_extractor',
        json_format='{role: str, goal: str, backstory: str, knowledge: str}'
    )
    config = extractor.run(inputs={'text': final_instructions})

    new_agent = create_agent(
        config['role'],
        config['goal'],
        config['backstory'],
        config['knowledge'],
    )
    log(color_text('✔ Novo agente criado', 'green'))

    # Fase 6: Teste do novo agente e cálculo de precisão
    log(color_text('--- Testando agente melhorado ---', 'blue'))
    test_scores = []
    for idx, (instance, expected) in enumerate(zip(training_data, expected_outputs), start=1):
        try:
            output = new_agent.run(inputs=instance)
            eval_resp = evaluator.run(inputs={
                'prompt': agent.get_description(),
                'input': instance,
                'output': output,
            })
            valid = validator.run(inputs={'evaluation': eval_resp}).get('result', 0)
            test_scores.append((instance, output, valid, expected))
            log(color_text(f"[{idx}] Resultado validado: {valid}", 'yellow'))
        except Exception as e:
            log(color_text(f"✖ Erro no teste da instância {idx}: {e}", 'red'))

    tp = sum(1 for (_, _, valid, exp), inst in zip(test_scores, results) if exp and valid == 1)
    fp = sum(1 for (_, _, valid, exp), inst in zip(test_scores, results) if not exp and valid == 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    print(color_text(f"Precision final: {precision:.2f}", 'bold'))

    # Fase 7: Salvando resultados
    util.write_all_text(
        'improved_results.txt',
        '\n\n'.join(
            f"{i+1}. {inst}\nExpected: {exp}\nGot: {out}" for i, (inst, out, _, exp) in enumerate(test_scores)
        )
    )
    util.write_all_text('config/improved_agent.yaml', final_instructions)

    return final_instructions

def ui_improve_gpt_prompt_by_ai(
    agent: GPTAgent,
    training_data: List[dict],
    trainer_model: str = default_model,
    *,
    verbose: bool = False,
    on_phase: Callable[[str], None] = None,
    on_step: Callable[[int, int], None] = None,
    **kwargs
) -> str:
    """
    Automatically improve a GPT agent’s prompt via AI-driven training, evaluation, and synthesis,
    with optional callbacks for phase updates and per-step progress.

    Parameters:
        agent (GPTAgent): The initial agent to optimize.
        training_data (List[dict]): Input instances + expected outputs.
        trainer_model (str): GPT model to use for training/evaluation.
        verbose (bool): If True, prints ANSI-colored logs to console.
        on_phase (callable): Called as on_phase(phase_name) at start of each phase.
        on_step  (callable): Called as on_step(idx, total) each loop in phase 1.
        **kwargs: Ignored.
    Returns:
        str: The final synthesized prompt instructions.
    """

    # ANSI colors for console (used if verbose=True)
    COLORS = {
        'reset': '\033[0m', 'bold': '\033[1m',
        'red': '\033[31m', 'green': '\033[32m',
        'yellow': '\033[33m', 'blue': '\033[34m',
    }
    def color_text(text: str, color: str = 'reset') -> str:
        return f"{COLORS.get(color, COLORS['reset'])}{text}{COLORS['reset']}"

    log = print if verbose else lambda *a, **k: None

    # Build evaluator
    evaluator = build('_trainer_evaluator')
    evaluator.gpt_model = trainer_model
    evaluator.max_tokens = default_max_tokens

    # Factory for trainers
    factory = GPTFactory()
    def make_trainer(
        name: str,
        max_tokens: int = default_max_tokens,
        temperature: float = None,
        json_format: str = None,
    ):
        t = factory.build(name)
        t.gpt_model = trainer_model
        t.max_tokens = max_tokens
        if temperature is not None:
            t.temperature = temperature
        if json_format is not None:
            t.json_format = json_format
        return t

    total = len(training_data)
    results = []

    # Phase 1: initial evaluation
    if on_phase: on_phase("Initial evaluation")
    for idx, instance in enumerate(training_data, start=1):
        if on_step: on_step(idx, total)
        log(color_text(f"[{idx}/{total}] Executing agent...", 'yellow'))
        try:
            output = agent.run(inputs=instance)
            eval_resp = evaluator.run(inputs={
                'prompt': agent.get_description(),
                'input': instance,
                'output': output,
            })
            results.append((instance, output, eval_resp))
            log(color_text("✔ Output successfully obtained", 'green'))
        except Exception as e:
            log(color_text(f"✖ Error on instance {idx}: {e}", 'red'))

    # Phase 2: generate fine-tuning instructions
    if on_phase: on_phase("Generating fine-tuning instructions")
    trainer = make_trainer('_trainer')
    instructions = []
    i = 0
    for instance, output, eval_resp in results:
        i += 1
        instr = trainer.run(inputs={
            'agent_description': agent.get_description(),
            'instance_data': instance,
            'result': output,
            'human_evaluation': eval_resp,
        })
        instructions.append(instr)
        if on_step: on_step(i, len(results))
    log(color_text("✔ Individual instructions generated", 'green'))

    # Phase 3: binary validation
    if on_phase: on_phase("Validating instructions")
    validator = make_trainer(
        '_trainer_binary_validator',
        max_tokens=200,
        temperature=0.2,
        json_format='{result: int}'
    )
    human_scores = [
        validator.run(inputs={'evaluation': instr}).get('result', 0)
        for instr in instructions
    ]
    log(color_text("✔ Validation complete", 'green'))

    # Phase 4: synthesize final instructions
    if on_phase: on_phase("Synthesizing final instructions")
    synthesizer = make_trainer('_trainer_synthesizer')
    final_instructions = synthesizer.run(inputs={
        'agent_description': agent.get_description(),
        'training_instructions': '\n'.join(instructions),
    })
    log(color_text("✔ Final instructions synthesized", 'green'))

    # Phase 5: extract new agent config
    if on_phase: on_phase("Extracting final configuration")
    extractor = make_trainer(
        '_json_extractor',
        json_format='{role: str, goal: str, backstory: str, knowledge: str}'
    )
    config = extractor.run(inputs={'text': final_instructions})
    new_agent = create_agent(
        config['role'],
        config['goal'],
        config['backstory'],
        config['knowledge'],
    )
    log(color_text("✔ New agent created", 'green'))

    # Phase 6: test improved agent
    if on_phase: on_phase("Testing improved agent")
    test_scores = []
    for idx, instance in enumerate(training_data, start=1):
        if on_step: on_step(idx, len(training_data))
        try:
            output = new_agent.run(inputs=instance)
            eval_resp = evaluator.run(inputs={
                'prompt': agent.get_description(),
                'input': instance,
                'output': output,
            })
            valid = validator.run(inputs={'evaluation': eval_resp}).get('result', 0)
            test_scores.append((instance, output, valid))
            log(color_text(f"[{idx}] Validated result: {valid}", 'yellow'))
        except Exception as e:
            log(color_text(f"✖ Error testing instance {idx}: {e}", 'red'))

    # Phase 7: calculating precision
    if on_phase: on_phase("Calculating precision")
    tp = sum(1 for hs, ts in zip(human_scores, test_scores) if hs == 1 and ts[2] == 1)
    fp = sum(1 for hs, ts in zip(human_scores, test_scores) if hs == 0 and ts[2] == 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    log(color_text(f"Final precision: {precision:.2f}", 'bold'))
    if on_phase: on_phase(f"Final precision: {precision:.2f}")

    # Phase 8: save results
    if on_phase: on_phase("Saving results")
    util.write_all_text(
        'improved_results.txt',
        '\n\n'.join(f"{i+1}. {inst}\n{out}" for i, (inst, out, _) in enumerate(test_scores))
    )
    util.write_all_text('config/improved_agent.yaml', final_instructions)

    if on_phase: on_phase(f"Final precision: {precision:.2f}")

    return final_instructions

def ui_improve_gpt_prompt_by_human(
    agent: GPTAgent,
    training_data: List[dict],
    trainer_model: str = default_model,
    *,
    verbose: bool = False,
    on_phase: Callable[[str], None] = None,
    on_step: Callable[[int, int], None] = None,
    **kwargs
) -> str:
    """
    Interactively improve a GPT agent’s prompt using human-in-the-loop evaluation and AI-driven synthesis,
    with optional callbacks for phase updates and per-step progress.
    """
    # ANSI colors para realce
    COLORS = {
        'reset': '\033[0m', 'bold': '\033[1m',
        'red': '\033[31m', 'green': '\033[32m',
        'yellow': '\033[33m', 'blue': '\033[34m',
    }
    def color_text(text: str, color: str = 'reset') -> str:
        return f"{COLORS.get(color, COLORS['reset'])}{text}{COLORS['reset']}"

    log = print if verbose else lambda *a, **k: None
    def _phase(name: str):
        if on_phase:
            on_phase(name)
        log(color_text(f"--- {name} ---", 'blue'))
    def _step(idx: int, total: int):
        if on_step:
            on_step(idx, total)

    # avaliador automático (usado na fase de teste)
    evaluator = build('_trainer_evaluator')
    evaluator.gpt_model = trainer_model
    evaluator.max_tokens = default_max_tokens

    factory = GPTFactory()
    def make_trainer(name: str, max_tokens: int = default_max_tokens,
                     temperature: float = None, json_format: str = None):
        t = factory.build(name)
        t.gpt_model = trainer_model
        t.max_tokens = max_tokens
        if temperature is not None:
            t.temperature = temperature
        if json_format is not None:
            t.json_format = json_format
        return t

    total = len(training_data)
    results = []

    # Fase 1: avaliação humana
    _phase("Avaliação humana via input()")
    for idx, instance in enumerate(training_data, start=1):
        _step(idx, total)
        log(color_text(f"[{idx}/{total}] Executando agente...", 'yellow'))
        try:
            output = agent.run(inputs=instance)
            print(color_text(f"Instância: {instance}", 'yellow'))
            print(color_text(f"Saída: {output}", 'yellow'))
            comment = input(color_text("Comentário sobre a saída: ", 'bold'))
            results.append((instance, output, comment))
            log(color_text(f"Comentário recebido: {comment}", 'green'))
        except Exception as e:
            log(color_text(f"✖ Erro na instância {idx}: {e}", 'red'))
    _phase("Avaliação humana concluída")

    # Fase 2: geração de instruções
    _phase("Gerando instruções de fine-tuning")
    trainer = make_trainer('_trainer')
    instructions = []
    for instance, output, human_comment in results:
        instr = trainer.run(inputs={
            'agent_description': agent.get_description(),
            'instance_data': instance,
            'result': output,
            'human_evaluation': human_comment,
        })
        instructions.append(instr)
    log(color_text('✔ Instruções individuais geradas', 'green'))

    # Fase 3: validação binária
    _phase("Validando instruções")
    validator = make_trainer(
        '_trainer_binary_validator',
        max_tokens=200,
        temperature=0.2,
        json_format='{result: int}'
    )
    human_scores = [
        validator.run(inputs={'evaluation': instr}).get('result', 0)
        for instr in instructions
    ]
    log(color_text('✔ Validação completa', 'green'))

    # Fase 4: síntese
    _phase("Sintetizando instruções finais")
    synthesizer = make_trainer('_trainer_synthesizer')
    final_instructions = synthesizer.run(inputs={
        'agent_description': agent.get_description(),
        'training_instructions': '\n'.join(instructions),
    })
    log(color_text('✔ Instruções finais sintetizadas', 'green'))

    # Fase 5: extração de config
    _phase("Extraindo configuração do agente melhorado")
    extractor = make_trainer(
        '_json_extractor',
        json_format='{role: str, goal: str, backstory: str, knowledge: str}'
    )
    config = extractor.run(inputs={'text': final_instructions})
    new_agent = create_agent(
        config['role'],
        config['goal'],
        config['backstory'],
        config['knowledge'],
    )
    log(color_text('✔ Novo agente criado', 'green'))

    # Fase 6: teste e cálculo de precisão
    _phase("Testando agente melhorado")
    test_scores = []
    for idx, instance in enumerate(training_data, start=1):
        _step(idx, total)
        try:
            output = new_agent.run(inputs=instance)
            eval_resp = evaluator.run(inputs={
                'prompt': agent.get_description(),
                'input': instance,
                'output': output,
            })
            valid = validator.run(inputs={'evaluation': eval_resp}).get('result', 0)
            test_scores.append((instance, output, valid))
            log(color_text(f"[{idx}] Resultado validado: {valid}", 'yellow'))
        except Exception as e:
            log(color_text(f"✖ Erro no teste da instância {idx}: {e}", 'red'))

    tp = sum(1 for gs, ts in zip(human_scores, test_scores) if gs == 1 and ts[2] == 1)
    fp = sum(1 for gs, ts in zip(human_scores, test_scores) if gs == 0 and ts[2] == 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    _phase(f"Precision final: {precision:.2f}")

    # Fase 7: salvando resultados
    _phase("Salvando resultados")
    util.write_all_text(
        'improved_results.txt',
        '\n\n'.join(
            f"{i+1}. {inst}\n{out}"
            for i, (inst, out, _) in enumerate(test_scores)
        )
    )
    util.write_all_text('config/improved_agent.yaml', final_instructions)

    return final_instructions

def ui_improve_gpt_prompt_by_data(
    agent: GPTAgent,
    training_data: List[dict],
    expected_outputs: List[str],
    trainer_model: str = default_model,
    *,
    verbose: bool = False,
    on_phase: Callable[[str], None] = None,
    on_step: Callable[[int, int], None] = None,
    **kwargs
) -> str:
    """
    Automatically improve a GPT agent’s prompt using data-driven training, evaluation, and synthesis,
    with optional callbacks for phase updates and per-step progress.
    """
    # ANSI colors para realce
    COLORS = {
        'reset': '\033[0m', 'bold': '\033[1m',
        'red': '\033[31m', 'green': '\033[32m',
        'yellow': '\033[33m', 'blue': '\033[34m',
    }
    def color_text(text: str, color: str = 'reset') -> str:
        return f"{COLORS.get(color, COLORS['reset'])}{text}{COLORS['reset']}"

    log = print if verbose else lambda *a, **k: None

    def _phase(name: str):
        if on_phase:
            on_phase(name)
        log(color_text(f"--- {name} ---", 'blue'))

    def _step(idx: int, total: int):
        if on_step:
            on_step(idx, total)

    # avaliador automático (usado na fase de teste)
    evaluator = build('_trainer_evaluator')
    evaluator.gpt_model = trainer_model
    evaluator.max_tokens = default_max_tokens

    factory = GPTFactory()
    def make_trainer(
        name: str,
        max_tokens: int = default_max_tokens,
        temperature: float = None,
        json_format: str = None,
    ):
        t = factory.build(name)
        t.gpt_model = trainer_model
        t.max_tokens = max_tokens
        if temperature is not None:
            t.temperature = temperature
        if json_format is not None:
            t.json_format = json_format
        return t

    total = len(training_data)
    results = []

    # Fase 1: execução e coleta de pares (output, expected)
    _phase("Avaliação por dados esperados")
    for idx, (instance, expected) in enumerate(zip(training_data, expected_outputs), start=1):
        _step(idx, total)
        log(color_text(f"[{idx}/{total}] Executando agente...", 'yellow'))
        try:
            output = agent.run(inputs=instance)
            log(color_text(f"Output: {output}", 'yellow'))
            log(color_text(f"Expected: {expected}", 'yellow'))
            results.append((instance, output, expected))
        except Exception as e:
            log(color_text(f"✖ Erro na instância {idx}: {e}", 'red'))
    _phase("Coleta de resultados concluída")

    # Fase 2: geração de instruções por instância
    _phase("Gerando instruções de fine-tuning")
    trainer = make_trainer('_trainer')
    instructions = []
    for instance, output, expected in results:
        instr = trainer.run(inputs={
            'agent_description': agent.get_description(),
            'instance_data': instance,
            'result': output,
            'human_evaluation': expected,
        })
        instructions.append(instr)
    log(color_text('✔ Instruções individuais geradas', 'green'))

    # Fase 3: validação binária das instruções
    _phase("Validando instruções")
    validator = make_trainer(
        '_trainer_binary_validator',
        max_tokens=200,
        temperature=0.2,
        json_format='{result: int}'
    )
    instr_scores = [
        validator.run(inputs={'evaluation': instr}).get('result', 0)
        for instr in instructions
    ]
    log(color_text('✔ Validação completa', 'green'))

    # Fase 4: síntese de instruções finais
    _phase("Sintetizando instruções finais")
    synthesizer = make_trainer('_trainer_synthesizer')
    final_instructions = synthesizer.run(inputs={
        'agent_description': agent.get_description(),
        'training_instructions': '\n'.join(instructions),
    })
    log(color_text('✔ Instruções finais sintetizadas', 'green'))

    # Fase 5: extração de novo config JSON
    _phase("Extraindo configuração do agente melhorado")
    extractor = make_trainer(
        '_json_extractor',
        json_format='{role: str, goal: str, backstory: str, knowledge: str}'
    )
    config = extractor.run(inputs={'text': final_instructions})
    new_agent = create_agent(
        config['role'],
        config['goal'],
        config['backstory'],
        config['knowledge'],
    )
    log(color_text('✔ Novo agente criado', 'green'))

    # Fase 6: teste do novo agente e cálculo de precisão
    _phase("Testando agente melhorado")
    test_scores = []
    for idx, (instance, expected) in enumerate(zip(training_data, expected_outputs), start=1):
        _step(idx, total)
        try:
            output = new_agent.run(inputs=instance)
            eval_resp = evaluator.run(inputs={
                'prompt': agent.get_description(),
                'input': instance,
                'output': output,
            })
            valid = validator.run(inputs={'evaluation': eval_resp}).get('result', 0)
            test_scores.append((instance, output, valid, expected))
            log(color_text(f"[{idx}] Resultado validado: {valid}", 'yellow'))
        except Exception as e:
            log(color_text(f"✖ Erro no teste da instância {idx}: {e}", 'red'))

    # cálculo de precisão
    tp = sum(1 for (_, _, valid, exp) in test_scores if exp and valid == 1)
    fp = sum(1 for (_, _, valid, exp) in test_scores if not exp and valid == 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    _phase(f"Precision final: {precision:.2f}")

    # Fase 7: salvando resultados
    _phase("Salvando resultados")
    util.write_all_text(
        'improved_results.txt',
        '\n\n'.join(
            f"{i+1}. {inst}\nExpected: {exp}\nGot: {out}"
            for i, (inst, out, _, exp) in enumerate(test_scores)
        )
    )
    util.write_all_text('config/improved_agent.yaml', final_instructions)

    return final_instructions