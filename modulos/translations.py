TEXTS = {
    "pt": {
        "about_text": (
            "Plataforma de Anotação de Vídeo\n\n"
            "Versão 1.0\n"
            "Desenvolvido para auxiliar na anotação de vídeos\n"
            "com detecção automática usando YOLO\n\n"
        ),

        # Títulos e textos principais
        "waiting_action": "Aguardando ação...",
        "no_loaded": "Nenhum vídeo carregado",
        "load_drag": "Arraste e solte um vídeo.",
        "create_dataset": "Criar dataset de treino",
        
        # Tooltips dos botões
        "load_video": "Carregar Vídeo",
        "detect_frame": "Detectar objetos no frame atual",
        "toggle_detection": "Ativar/desativar detecção contínua",
        "annotate_manual": "Anotação manual",
        "save_annotations": "Salvar anotações",
        "save_frame": "Salvar frame com anotações",
        "live": "Modo ao vivo",
        "remove_selected_taxons": "Remover táxons selecionados",
        "confirm_remove_title": "Remover táxons",
        "confirm_remove_multiple": "Remover {} táxons selecionados?",
        "merge_annotations": "Mesclar anotações (georreferenciamento)",
        
        # Menu Arquivo
        "arquivo": "Arquivo",
        "load_model": "Carregar Modelo",
        "unload_model": "Descarregar Modelo",
        "load_annotations": "Carregar Anotações",
        "export_yolo": "Exportar para YOLO",
        "start_recording": "Iniciar Gravação",
        "stop_recording": "Parar Gravação",
        "exit": "Sair",
        
        # Menu Visualização
        "visualization": "Visualização",
        "history_show": "Mostrar/Ocultar Histórico",
        "dark_mode": "Modo Escuro",
        
        # Menu Anotação
        "annotation": "Anotação",
        "train_yolo": "Treinar Modelo YOLO",
        
        # Menu Idioma
        "language": "Idioma",
        "portuguese": "Português",
        "english": "Inglês",
        
        # Menu Ajuda
        "help": "Ajuda",
        "shortcuts": "Atalhos do Teclado",
        "about": "Sobre",
        
        # Histórico de detecções
        "history": "Histórico de Detecções",
        "all_classes": "Todas os táxons",
        "filter_detection": "Filtrar Detecções",
        "filter": "Filtrar",
        "taxon": "Táxon:",
        "taxons": "Táxons",
        "confidence": "Confiança:",
        
        # Mensagens de status
        "model_loaded": "Modelo carregado: {}",
        "webcam_mode_off": "Modo webcam desativado",
        "webcam_mode_on": "Modo webcam ativado (Câmera {})",
        "recording_started": "Gravando: {}",
        "recording_stopped": "Gravação finalizada: {}",
        "no_recording_to_save": "Nenhuma gravação para salvar",
        "video_saved": "Vídeo salvo em: {}",
        "video_annotations_saved": "Vídeo e anotações salvos em: {}",
        "video_load_error": "Erro ao abrir o vídeo",
        "video_first_frame_error": "Erro ao ler o primeiro frame do vídeo",
        "camera_error": "Erro ao iniciar câmera: {}",
        "detection_completed": "Detecção concluída no frame {}",
        "detection_error": "Erro na detecção: {}",
        "no_video_loaded": "Nenhum vídeo carregado!",
        "no_model_loaded": "Nenhum modelo carregado!",
        "continuous_detection_on": "Detecção contínua ATIVADA",
        "continuous_detection_off": "Detecção contínua DESATIVADA",
        "manual_annotation_off": "Anotação manual DESATIVADA",
        "fatal_error": "Erro grave - reinicie a aplicação",
        "annotations_saved": "Anotações salvas como CSV em: {}",
        "annotations_loaded": "Anotações carregadas de {}",
        "annotations_loaded_with_model": "Anotações carregadas de {}. Modelo usado: {}",
        "frame_saved": "Frame salvo como: {}",
        "frames_saved": "Frames salvos em: {}",
        "no_frame_to_save": "Nenhum frame para salvar",
        "velocity_1x": "Velocidade: 1.0x (FPS: {:.1f})",
        "velocity_2x": "Velocidade: 2.0x (FPS: {:.1f})",
        "velocity_2x_detection": "Velocidade: 2.0x (Detecção a cada {} frames)",
        "saving_error": "Erro ao salvar anotações",
        "saving_video_error": "Erro ao salvar vídeo: {}",
        
        # Diálogos e mensagens
        "warning": "Aviso",
        "error": "Erro",
        "select_class": "Selecionar Classe",
        "select_or_type_class": "Selecione ou digite uma nova classe:",
        "select_camera": "Selecionar Câmera",
        "choose_camera": "Escolha a câmera:",
        "select_annotations_file": "Selecionar Arquivo de Anotações",
        "select_georeferencing_file": "Selecionar Arquivo de Georreferenciamento",
        "select_model": "Selecionar Modelo YOLO",
        "select_video": "Selecionar Vídeo",
        "save_recording_title": "Salvar Gravação",
        "save_recording_question": "Gravação concluída. Deseja salvar o vídeo?",
        "save_annotations_dialog": "Salvar Anotações",
        "save_frame_dialog": "Salvar Frame",
        "export_yolo_dialog": "Selecionar Diretório para Exportar Anotações YOLO",
        "train_dataset_dialog": "Selecionar Diretório para Dataset de Treinamento",
        "advanced_training_settings": "Configurações Avançadas de Treinamento",
        "exporting_dataset": "Exportando Dataset YOLO",
        "training_model": "Treinamento do Modelo YOLO",
        "keyboard_shortcuts": "Atalhos do Teclado",
        "about_title": "Sobre",
        "navigate_frames": "Navegar entre frames",
        "choose_merge_columns": "Escolha as colunas-chave",
        "key_column_left": "Coluna no CSV de anotações",
        "key_column_right": "Coluna no CSV de georreferenciamento",
        "choose_merge_column": "Escolha a coluna para unir",
        "save_merged_annotations": "Salvar anotações georreferenciadas",
        "merge_completed": "Merge concluído!\nArquivo salvo em: {}",
        "merge_error": "Erro ao fazer merge: {}",
        "debug_merge_error": "Erro no merge: {}",
        "success": "Sucesso",
        "merge_completed": "Merge concluído!\nArquivo salvo em: {}",
                
        # Textos de treinamento e exportação
        "no_annotations_to_export": "Nenhuma anotação disponível para exportar.",
        "no_manual_annotations": "Nenhuma anotação manual válida encontrada.",
        "no_manual_annotations_train": "Nenhuma anotação manual disponível para treinamento.",
        "export_completed": "Exportação Concluída",
        "training_completed": "Treinamento Concluído",
        "training_progress": "Treinando...",
        "training_success": (
            "Treinamento concluído com sucesso!\n\n"
            "Modelo salvo em:\n{}\n\n"
            "Deseja carregar o novo modelo?"
        ),
        "training_error": "Falha no treinamento: {}",
        "load_new_model_question": "Deseja carregar o novo modelo?",
        "cancel": "Cancelar",
        "training_error_title": "Erro no Treinamento",
        "debug_annotation_frame_error": "Erro processando anotação no frame {}: {}",
        
        # Configurações de treinamento
        "epochs": "Épocas:",
        "batch_size": "Batch:",
        "image_size": "Tamanho da imagem:",
        "learning_rate": "Taxa de aprendizado:",
        "device": "Dispositivo:",
        "config_failed": "Falha ao configurar treinamento: {}",
        "debug_config_failed": "Erro detalhado: {}",

        #Anotação manual
        "annotation_deleted": "Anotação excluída",
        "annotation_duplicated": "Anotação duplicada",
        "annotation_too_small": "Anotação muito pequena",
        "no_annotation_selected": "Nenhuma anotação selecionada",
        
        # Outros
        "exporting_frames": "Exportando frames...",
        "preparing_training": "Preparando treinamento...",
        "recording": "Gravando",
        "live_text": "Live",
        "model_unloaded": "Modelo descarregado",
        "invalid_annotations": "Arquivo de anotações inválido - formato não reconhecido",
        "previous": "Anterior",
        "pause": "Pausar", 
        "next": "Próximo",
        "play": "Reproduzir",
        "webcam": "Webcam (Câmera {})",
        "model_load_error": "Falha ao carregar modelo: {}",
        "video_loaded": "Vídeo carregado. Pressione Play para iniciar.",
        "recording_start_error": "Erro ao iniciar gravação",
        "error_colon": "Erro: {}",
        "error_reading_frame": "Erro ao ler o quadro do vídeo.",
        "no_model_loaded": "Nenhum modelo carregado.",
        "no_model_loaded_error": "Erro: Nenhum modelo carregado!",
        "video_name_format": "Vídeo: {}",
        "fatal_error": "Erro grave - reinicie a aplicação",
        "fatal_error_detail": "Ocorreu um erro grave:\n{}\n\nDetalhes no console.",
        "speed_format": "Velocidade: {}x (FPS: {:.1f})",
        "speed_detection_format": "Velocidade: {}x (Detecção a cada {} frames)",
        "manual_annotation_off": "Anotação manual DESATIVADA",
        "export_completed": "Exportação Concluída",
        "background_images_added" : "Adicionadas {0} imagens de fundo ao dataset.",
        "export_success": (
            "Dataset exportado com sucesso!\n\n"
            "Frames processados: {}/{}\n"
            "Anotações totais: {}\n"
            "Classes: {}\n"
            "Background images: {}\n"
            "Local: {}"),
        "export_error": ("Erro na exportação\n"
                              "Ocorreu um erro: {}\n\n"
                              "Detalhes técnicos:{}"
        ),
        "load_annotations_error": "Falha ao carregar anotações:\n{}",
        "load_annotations_status_error": "Erro ao carregar anotações: {}",
        "debug_load_annotations_error": "Erro ao carregar anotações: {}",
        "frame_error": "Erro ao exibir frame: {}",
        "debug_fatal_update_frame": "Erro fatal no update_frame: {}",
        "model_not_found": "Arquivo não encontrado: {}",
        "camera_open_failed": "Não foi possível abrir a câmera {}",
        "camera_name": "Câmera {}",
        "name_model_title": "Nome do modelo",
        "name_model_label": "Digite um nome para o modelo treinado:",
        "manual_annotation_on": "Anotação manual ativada",
        "manual_annotation_off": "Anotação manual desativada",
        "add_taxon": "+ Adicionar táxon",
        "taxon_name": "Nome do táxon:",
        "new_taxon": "Novo táxon",
        "confirm_deletion": "Confirmar exclusão",
        "deletion_question": "Deseja excluir esta detecção?",
        "delete_detection": "Excluir detecção",
        "train": "Treino",
        "image_source": "Origem das imagens",
        "load_photos": "Carregar fotos",
        "frames_found": "Frames encontrados:",
        "extraction_rate": "Taxa de extração",
        "extract_every_n": "Extrair 1 frame a cada N:",
        "select_output_dir": "Pasta para guardar frames",
        "extracting_frames": "Extraindo frames…",
        "select_photos": "Selecione as fotos",
        "no_images_loaded_warning": "Nenhuma imagem foi carregada.",
        "dataset_ready": "Dataset pronto",
        "frames_loaded": "{num_frames} frames carregados.",
        "annotation_instruction": "Anota frame-a-frame e depois use Exportar para YOLO ou Treinar Modelo YOLO.",
        "space": "Espaço"

    },
    "en": {
        "about_text": (
            "Video Annotation Platform\n\n"
            "Version 1.0\n"
            "Developed to assist in video annotation\n"
            "with automatic detection using YOLO\n\n"
        ),
        
        # Títulos e textos principais
        "waiting_action": "Waiting for action...",
        "no_loaded": "No video loaded",
        "load_drag": "Drag and drop a video.",
        "create_dataset": "Create training dataset",
        
        # Tooltips dos botões
        "load_video": "Load Video",
        "detect_frame": "Detect objects in current frame",
        "toggle_detection": "Toggle continuous detection",
        "annotate_manual": "Manual annotation",
        "save_annotations": "Save annotations",
        "save_frame": "Save frame with annotations",
        "live": "Live mode",
        "remove_selected_taxons": "Remove the selected taxons",
        "confirm_remove_title": "Remove taxons",
        "confirm_remove_multiple": "Remove {} selected taxons?",
        "merge_annotations": "Merge annotations (georeferencing)",
            
        # Menu Arquivo
        "arquivo": "File",
        "load_model": "Load Model",
        "unload_model": "Unload Model",
        "load_annotations": "Load Annotations",
        "export_yolo": "Export to YOLO",
        "start_recording": "Start Recording",
        "stop_recording": "Stop Recording",
        "exit": "Exit",
        
        # Menu Visualização
        "visualization": "View",
        "history_show": "Show/Hide History",
        "dark_mode": "Dark Mode",
        
        # Menu Anotação
        "annotation": "Annotation",
        "train_yolo": "Train YOLO Model",
        
        # Menu Idioma
        "language": "Language",
        "portuguese": "Portuguese",
        "english": "English",
        
        # Menu Ajuda
        "help": "Help",
        "shortcuts": "Keyboard Shortcuts",
        "about": "About",

        
        # Histórico de detecções
        "history": "Detection History",
        "all_classes": "All taxons",
        "filter_detection": "Filter Detections",
        "filter": "Filter",
        "taxon": "Taxon:",
        "taxons": "Taxons",
        "confidence": "Confidence",
        
        # Mensagens de status
        "model_loaded": "Model loaded: {}",
        "webcam_mode_off": "Webcam mode deactivated",
        "webcam_mode_on": "Webcam mode activated (Camera {})",
        "recording_started": "Recording: {}",
        "recording_stopped": "Recording finished: {}",
        "no_recording_to_save": "No recording to save",
        "video_saved": "Video saved at: {}",
        "video_annotations_saved": "Video and annotations saved at: {}",
        "video_load_error": "Error opening video",
        "video_first_frame_error": "Error reading first video frame",
        "camera_error": "Error starting camera: {}",
        "detection_completed": "Detection completed on frame {}",
        "detection_error": "Detection error: {}",
        "no_video_loaded": "No video loaded!",
        "no_model_loaded": "No model loaded!",
        "continuous_detection_on": "Continuous detection ACTIVATED",
        "continuous_detection_off": "Continuous detection DEACTIVATED",
        "manual_annotation_off": "Manual annotation DEACTIVATED",
        "fatal_error": "Fatal error - restart the application",
        "annotations_saved": "Annotations saved as CSV at: {}",
        "annotations_loaded": "Annotations loaded from {}",
        "annotations_loaded_with_model": "Annotations loaded from {}. Model used: {}",
        "frame_saved": "Frame saved as: {}",
        "frames_saved": "Frames saved to: {}",
        "no_frame_to_save": "No frame to save",
        "velocity_1x": "Speed: 1.0x (FPS: {:.1f})",
        "velocity_2x": "Speed: 2.0x (FPS: {:.1f})",
        "velocity_2x_detection": "Speed: 2.0x (Detection every {} frames)",
        "saving_error": "Error saving annotations",
        "saving_video_error": "Error saving video: {}",
        
        # Diálogos e mensagens
        "warning": "Warning",
        "error": "Error",
        "select_class": "Select Class",
        "select_or_type_class": "Select or type a new class:",
        "select_camera": "Select Camera",
        "choose_camera": "Choose camera:",
        "select_annotations_file": "Select Annotation File",
        "select_georeferencing_file": "Select Georegerencing File",
        "select_model": "Select YOLO Model",
        "select_video": "Select Video",
        "save_recording_title": "Save Recording",
        "save_recording_question": "Recording completed. Do you want to save the video?",
        "save_annotations_dialog": "Save Annotations",
        "save_frame_dialog": "Save Frame",
        "export_yolo_dialog": "Select Directory to Export YOLO Annotations",
        "train_dataset_dialog": "Select Directory for Training Dataset",
        "advanced_training_settings": "Advanced Training Settings",
        "exporting_dataset": "Exporting YOLO Dataset",
        "training_model": "YOLO Model Training",
        "keyboard_shortcuts": "Keyboard Shortcuts",
        "about_title": "About",
        "navigate_frames": "Navigate between frames",
        "choose_merge_columns": "Choose key columns",
        "key_column_left": "Column in annotation CSV",
        "key_column_right": "Column in georeferencing CSV",
        "choose_merge_column": "Choose the column to merge",
        "save_merged_annotations": "Save georeferenced annotations",
        "merge_completed": "Merge completed!\nFile saved at: {}",
        "merge_error": "Merge error: {}",
        "debug_merge_error": "Merge error: {}",
        "success": "Sucesso",
        "merge_completed": "Merge concluído!\nArquivo salvo em: {}",
        
        # Textos de treinamento e exportação
        "no_annotations_to_export": "No annotations available to export.",
        "no_manual_annotations": "No valid manual annotations found.",
        "no_manual_annotations_train": "No manual annotations available for training.",
        "export_completed": "Export Completed",
        "training_completed": "Training Completed",
        "training_progress": "Training...",
        "training_success": (
            "Training completed successfully!\n\n"
            "Model saved at:\n{}\n\n"
            "Do you want to load the new model?"
        ),
        "training_error": "Training failed:",
        "load_new_model_question": "Do you want to load the new model?",
        "cancel": "Cancel",
        "training_error_title": "Training Error",
        "debug_annotation_frame_error": "Error processing annotation on frame {}: {}",
        
        # Configurações de treinamento
        "epochs": "Epochs:",
        "batch_size": "Batch size:",
        "image_size": "Image size:",
        "learning_rate": "Learning rate:",
        "device": "Device:",
        "config_failed": "Failed to configure training: {}",
        "debug_config_failed": "Detailed error: {}",

        #Anotação manual
        "annotation_deleted": "Annotation deleted",
        "annotation_duplicated": "Annotation duplicated",
        "annotation_too_small": "Annotation too small",
        "no_annotation_selected": "No annotation selected",
        
        # Outros
        "exporting_frames": "Exporting frames...",
        "preparing_training": "Preparing training...",
        "recording": "Recording",
        "live_text": "Live",
        "model_unloaded": "Model unloaded",
        "invalid_annotations": "Invalid annotations file - format not recognized",
        "previous": "Previous",
        "pause": "Pause",
        "next": "Next", 
        "play": "Play",
        "webcam": "Webcam (Camera {})",
        "model_load_error": "Failed to load model: {}",
        "video_loaded": "Video loaded. Press Play to start.",
        "recording_start_error": "Error starting recording",
        "error_colon": "Error: {}",
        "error_reading_frame": "Error reading video frame.",
        "no_model_loaded": "No model loaded.",
        "no_model_loaded_error": "Error: No model loaded!",
        "video_name_format": "Video: {}",
        "fatal_error": "Fatal error - restart the application",
        "fatal_error_detail": "A fatal error occurred:\n{}\n\nDetails in console.",
        "speed_format": "Speed: {}x (FPS: {:.1f})",
        "speed_detection_format": "Speed: {}x (Detection every {} frames)",
        "manual_annotation_off": "Manual annotation DEACTIVATED",
        "export_completed" : "Export Completed",
        "background_images_added" : "{0} background images added to the dataset.",
        "export_success": (
            "Dataset exported successfully!\n\n"
            "Processed frames: {}/{}\n"
            "Total annotations: {}\n"
            "Classes: {}\n"
            "Imagens de fundo: {}\n"
            "Location: {}"
        ),
         "export_error": ("Export error\n"
                              "An error occurred: {}\n\n"
                              "Technical details:{}"
        ),
        "load_annotations_error": "Failed to load annotations:\n{}",
        "load_annotations_status_error": "Error loading annotations: {}",
        "debug_load_annotations_error": "Error loading annotations: {}",
        "frame_error": "Error displaying frame: {}",
        "debug_fatal_update_frame": "Fatal error in update_frame: {}",
        "model_not_found": "File not found: {}",
        "camera_open_failed": "Could not open camera {}",
        "camera_name": "Camera {}",
        "name_model_title": "Model name",
        "name_model_label": "Enter a name for the trained model:",
        "manual_annotation_on": "Manual annotation enabled",
        "manual_annotation_off": "Manual annotation disabled",
        "add_taxon": "+ Add taxon",
        "taxon_name": "Name of the taxon:",
        "new_taxon": "New taxon",
        "confirm_deletion": "Confirm deletion",
        "deletion_question": "Do you want to delete this detection?",
        "delete_detection": "Delete detection",
        "train": "Train",
        "image_source": "Image source",
        "load_photos": "Load photos",
        "frames_found": "Frames found:",
        "extraction_rate": "Extraction rate",
        "extract_every_n": "Extract 1 frame every N:",
        "select_output_dir": "Folder to save frames",
        "extracting_frames": "Extracting frames…",
        "select_photos": "Select photos",
        "no_images_loaded_warning": "No images were loaded.",
        "dataset_ready": "Dataset Ready",
        "frames_loaded": "{num_frames} frames loaded.",
        "annotation_instruction": "Annotate frame-by-frame and then use Export to YOLO or Train YOLO Model.",
        "space": "Space"
    }
}