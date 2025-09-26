 CREATE DATABASE vsaas_postsearch
     WITH
     OWNER = vsaas_db
     ENCODING = 'UTF8'
     LC_COLLATE = 'en_US.UTF-8'
     LC_CTYPE = 'en_US.UTF-8'
     TABLESPACE = pg_default
     CONNECTION LIMIT = -1;


CREATE TABLE public.VERSION
(
    id serial PRIMARY KEY,
    version text,
    sql_cmd text,
    success bool default FALSE,
    update_time timestamp without time zone DEFAULT now()
);

CREATE VIEW LATEST_VERSION AS SELECT id, version, sql_cmd, success, update_time FROM VERSION WHERE id=( SELECT max(id) FROM version );
INSERT INTO DUMMY (test) VALUES(0);
INSERT INTO VERSION (version, sql_cmd) VALUES('20230101_01', 'SELECT * FROM current_catalog;');
/*# 20220427: Remove elk_index_prefix text,*/
CREATE TABLE public.CAMERA_INFO
(   
    mac varchar(36) NOT NULL,
    thingname varchar (56),
    last_report_time timestamp without time zone, 
    group_id text,
    organization_id text,
    partition_group_id int default -1::integer,
    model_name varchar(30), 
    ip varchar(15),
    timezone varchar(50),
    license_period_in_day int default 30,
    UNIQUE(mac, partition_group_id), 
    report_version text,
    face_version text,
    thumbnail_version text,
    CONSTRAINT pk_camera_info PRIMARY KEY (mac)
) TABLESPACE pg_default;
CREATE INDEX idx_camera_info_group_id ON CAMERA_INFO(group_id);
CREATE INDEX idx_camera_info_organization_id ON CAMERA_INFO(organization_id);

CREATE TABLE public.MODEL_HISTORY
(
    id serial,
    first_time timestamp without time zone NOT NULL,
    last_time timestamp without time zone,
    target text, -- T for thumbnail and F for face
    size_w int default 0,
    size_h int default 0,
    size_d int,
    version text,
    s3_bucket text,
    s3_key text,
    CONSTRAINT pk_model_history PRIMARY KEY (version, target)
);

CREATE TABLE public.CAMERA_MODEL_HISTORY
(
    id serial,
    mac text,
    version text,
    target text, -- T for thumbnail and F for face    
    update_time timestamp without time zone DEFAULT now(),
    CONSTRAINT pk_camera_model_history PRIMARY KEY (mac, version, target)
);

CREATE TABLE public.PARTITION_INFO
(
    id serial,
    partition_id varchar(16),
    mac text,
    tbl_name_object_trace text,
    last_update date,
    CONSTRAINT pk_partition_info PRIMARY KEY (partition_id, mac),
    CONSTRAINT fk_partition_info_camera_info FOREIGN KEY(mac)
    REFERENCES CAMERA_INFO(mac) ON UPDATE CASCADE ON DELETE CASCADE
);
CREATE INDEX idx_partion_info_lastupdate ON PARTITION_INFO(last_update);
CREATE INDEX idx_partion_info_mac_lastupdate ON PARTITION_INFO(mac, last_update);
CREATE INDEX idx_partion_info_partition_id ON PARTITION_INFO(partition_id);
CREATE INDEX idx_partion_info_id ON PARTITION_INFO(id);

CREATE TABLE public.OBJECT_TRACE
(
    partition_id varchar(16),
    oid bigint,
    mac character varying(48) NOT NULL,
    obj_type character varying(30),
    obj_attribute jsonb,
    first_time timestamp without time zone,
    obj_trace geometry(LineStringZ),
    thumbnail_json jsonb,
    relate bigint,
    hiding_info_json jsonb,
    has_face bool DEFAULT false,
    invalidate bool DEFAULT false,
    extra_info jsonb,
    CONSTRAINT pk_object_trace PRIMARY KEY (oid, mac, partition_id)    
) partition by list (partition_id);
CREATE INDEX idx_oid_object_trace ON OBJECT_TRACE(oid);
CREATE INDEX idx_mac_object_trace ON OBJECT_TRACE(mac);
CREATE INDEX idx_invalidate ON OBJECT_TRACE(invalidate);
CREATE INDEX idx_obj_type_object_trace ON OBJECT_TRACE(obj_type);
CREATE INDEX idx_obj_trace_first_time_object_trace ON OBJECT_TRACE(first_time);
CREATE INDEX idx_obj_attribute_object_trace ON OBJECT_TRACE USING GIN(obj_attribute jsonb_path_ops);
CREATE INDEX idx_partitionid_object_trace ON OBJECT_TRACE(partition_id);

CREATE TABLE public.FACE_PROFILE
(
    id serial,
    name text,
    author text,
    organization text,
    tag varchar[],
    description text,
    create_time timestamp without time zone DEFAULT now(),
    last_update_time timestamp without time zone DEFAULT now(),
    CONSTRAINT pk_face_profile PRIMARY KEY (id)
);

CREATE TABLE public.FACE_PROFILE_OBJECT
(
    id serial,
    oid bigint,
    mac text,
    profile_id bigint,    
    thumbnail_json jsonb,
    author text,
    img_s3_bucket text,
    img_s3_key text,
    img_s3_start int default 0,
    img_s3_length int default 0,
    is_cover bool,
    cluster int NOT NULL DEFAULT 0,    
    last_update_time timestamp without time zone DEFAULT now(),
    CONSTRAINT pk_face_profile_object PRIMARY KEY (id),
    CONSTRAINT uk_face_profile_object UNIQUE (oid,mac),
    CONSTRAINT fk_face_profile_object_face_profile FOREIGN KEY(profile_id)
    REFERENCES FACE_PROFILE(id) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE public.FACE_PROFILE_CLUSTER_MODEL_CACHE
(
    id serial,
    profile_id bigint,
    version text,
    cluster int,
    vector float[],
    last_update_time timestamp without time zone DEFAULT now(),
    CONSTRAINT pk_face_profile_cluster_model_cache PRIMARY KEY (profile_id, version, cluster),
    CONSTRAINT fk_face_profile_cluster_model_cache_face_profile FOREIGN KEY(profile_id)
    REFERENCES FACE_PROFILE(id) ON UPDATE CASCADE ON DELETE CASCADE
    
);

CREATE TABLE public.FACE_PROFILE_OBJECT_FEEDBACK
(
    id serial,
    oid bigint,
    mac text,
    profile_id bigint,
    thumbnail_json jsonb,
    author text,
    img_s3_bucket text,
    img_s3_key text,
    img_s3_start int default 0,
    img_s3_length int default 0,
    cluster int NOT NULL DEFAULT 0,
    last_update_time timestamp without time zone DEFAULT now(),
    CONSTRAINT pk_face_profile_object_feedback PRIMARY KEY (id),
    CONSTRAINT uk_face_profile_object_feedback UNIQUE (oid,mac),
    CONSTRAINT fk_face_profile_object_feedback_face_profile FOREIGN KEY(profile_id)
    REFERENCES FACE_PROFILE(id) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE public.FACE_PROFILE_FEEDBACK_CLUSTER_MODEL_CACHE
(
    id serial,
    profile_id bigint,
    version text,
    cluster int,
    vector float[],
    last_update_time timestamp without time zone DEFAULT now(),
    CONSTRAINT pk_face_profile_feedback_cluster_model_cache PRIMARY KEY (profile_id, version, cluster),
    CONSTRAINT fk_face_profile_feedback_cluster_model_cache_face_profile FOREIGN KEY(profile_id)
    REFERENCES FACE_PROFILE(id) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS public.FACE_PROFILE_DELETE_RECORD
(
    id serial,
    profile_id bigint,
    organization text,
    last_update_time timestamp without time zone DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_face_profile_delete_record_organization_last_update_time ON FACE_PROFILE_DELETE_RECORD(organization, last_update_time);

CREATE TABLE public.CAMERA_LOG
(
    id serial,
    mac character varying(48) NOT NULL,
    day date, /* the day of counting */
    count int, /* the number of objects */    
    CONSTRAINT pk_camera_log PRIMARY KEY (mac, day),
    CONSTRAINT camera_log_mac_fkey FOREIGN KEY( mac )
    REFERENCES CAMERA_INFO( mac )  ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS public.FACE_PROFILE_OBJECT_MODEL_CACHE
(
    id serial,
    face_profile_object_id bigint,
    version text,
    vector float[],
    last_update_time timestamp without time zone DEFAULT now(),
    CONSTRAINT pk_face_profile_object_model_cache PRIMARY KEY (id),
    CONSTRAINT uk_face_profile_object_model_cache UNIQUE(face_profile_object_id, version),
    CONSTRAINT fk_face_profile_object_model_cache_face_profile_object FOREIGN KEY(face_profile_object_id)
    REFERENCES FACE_PROFILE_OBJECT(id) ON UPDATE CASCADE ON DELETE CASCADE
);

ALTER TABLE public.VERSION
    OWNER to vivotek;
ALTER TABLE public.CAMERA_INFO
    OWNER to vivotek;
ALTER TABLE public.OBJECT_TRACE
    OWNER to vivotek;
ALTER TABLE public.PARTITION_INFO
    OWNER to vivotek;
ALTER TABLE public.FACE_PROFILE
    OWNER to vivotek;
ALTER TABLE public.FACE_PROFILE_OBJECT
    OWNER to vivotek;
ALTER TABLE public.FACE_PROFILE_FEEDBACK_CLUSTER_MODEL_CACHE
    OWNER to vivotek;
ALTER TABLE public.CAMERA_LOG
    OWNER to vivotek;

CREATE OR REPLACE FUNCTION insert_new_camera()
    RETURNS TRIGGER 
    LANGUAGE PLPGSQL
    AS
    $$
    DECLARE
    -- 100  cameras in one group for PG
    -- 1000 cameras in one group for ELK
    MAX_GROUP_SIZE constant int := 100;    
    max_partition_group_id int;
    global_max_partition_group_id int;
    partition_group_size int;

    BEGIN
        -- Negative partition_id means to generate it automatically.
        IF NEW.group_id is not null OR char_length(NEW.group_id)>0 THEN
            SELECT COALESCE(max(partition_group_id),0) INTO global_max_partition_group_id FROM CAMERA_INFO;        
            SELECT COALESCE(max(partition_group_id),0) INTO max_partition_group_id FROM CAMERA_INFO WHERE group_id=NEW.group_id;
            SELECT count(mac) INTO partition_group_size FROM CAMERA_INFO WHERE partition_group_id=max_partition_group_id;
            IF max_partition_group_id=0 THEN
                max_partition_group_id=global_max_partition_group_id+1;
            END IF;
            IF partition_group_size>=MAX_GROUP_SIZE THEN
                max_partition_group_id=global_max_partition_group_id+1;
            END IF;
            NEW.partition_group_id=max_partition_group_id;
        ELSE
            NEW.partition_group_id=0;
        END IF;        
        RETURN NEW;
    END
    $$;
CREATE TRIGGER camera_info_insert_audit_trig BEFORE INSERT ON public.CAMERA_INFO
FOR EACH ROW EXECUTE PROCEDURE insert_new_camera();

CREATE OR REPLACE FUNCTION update_camera()
    RETURNS TRIGGER 
    LANGUAGE PLPGSQL
    AS
    $$
    DECLARE
    -- 100  cameras in one group for PG
    -- 1000 cameras in one group for ELK
    MAX_GROUP_SIZE constant int := 100;    
    max_partition_group_id int;
    global_max_partition_group_id int;
    partition_group_size int;

    BEGIN
        -- Negative partition_id means to generate it automatically.
        IF NEW.group_id is not null or char_length(NEW.group_id)>0 THEN
            SELECT COALESCE(max(partition_group_id),0) INTO global_max_partition_group_id FROM CAMERA_INFO;        
            SELECT COALESCE(max(partition_group_id),0) INTO max_partition_group_id FROM CAMERA_INFO WHERE group_id=NEW.group_id;
            SELECT count(mac) INTO partition_group_size FROM CAMERA_INFO WHERE partition_group_id=max_partition_group_id;
            IF max_partition_group_id=0 THEN
                max_partition_group_id=global_max_partition_group_id+1;
            END IF;
            IF partition_group_size>=MAX_GROUP_SIZE THEN
                max_partition_group_id=global_max_partition_group_id+1;
            END IF;
            NEW.partition_group_id=max_partition_group_id;
        END IF;       
        RETURN NEW;
    END
    $$;

CREATE TRIGGER camera_info_update_audit_trig BEFORE UPDATE OF group_id ON public.CAMERA_INFO
FOR EACH ROW EXECUTE PROCEDURE update_camera();


CREATE OR REPLACE FUNCTION public.insert_new_data() 
    RETURNS TRIGGER
    LANGUAGE PLPGSQL
    AS $$
    BEGIN
        IF NEW.insertion_time IS NULL THEN
            SELECT NOW() INTO NEW.insertion_time;
        END IF;
        RETURN NEW;
    END
    $$;

CREATE OR REPLACE FUNCTION insert_new_partition()
-- Put null partition_id to generate by this script.
    RETURNS TRIGGER 
    LANGUAGE PLPGSQL
    AS
    $$
    DECLARE
    date_today text;
    month_today text;
    year_today text;
    _partition_id text;
    _partition_group_id int;
    is_old_partition bool;
    _tbl_name_object_trace text;
    BEGIN
        IF NEW.last_update IS NULL THEN
            SELECT DATE_PART('day',NOW()) into date_today;
            SELECT DATE_PART('month',NOW()) into month_today;
            SELECT DATE_PART('year',NOW()) into year_today;
            SELECT NOW() INTO NEW.last_update;
        ELSE
            SELECT DATE_PART('day',NEW.last_update) into date_today;
            SELECT DATE_PART('month',NEW.last_update) into month_today;
            SELECT DATE_PART('year',NEW.last_update) into year_today;             
        END IF;
        SELECT CAMERA_INFO.partition_group_id  INTO _partition_group_id FROM CAMERA_INFO WHERE mac=NEW.mac;
        
        SELECT format('%s_%s_%s_%s', _partition_group_id, year_today, month_today, date_today) INTO _partition_id;        
        SELECT format('%s_%s', 'OBJECT_TRACE', _partition_id) INTO _tbl_name_object_trace;

        SELECT exists(SELECT PI.partition_id FROM PARTITION_INFO AS PI WHERE PI.partition_id=_partition_id) into is_old_partition;
        IF is_old_partition IS FALSE THEN
            -- create partition
            execute format('CREATE TABLE IF NOT EXISTS "%s" PARTITION OF OBJECT_TRACE FOR VALUES IN (''%s'');',_tbl_name_object_trace,_partition_id);
           
        END IF;
        NEW.tbl_name_object_trace=_tbl_name_object_trace;
        NEW.partition_id=_partition_id;
        RETURN NEW;
    END
    $$;

CREATE TRIGGER partition_info_insert_audit_trig BEFORE INSERT ON public.PARTITION_INFO
FOR EACH ROW EXECUTE PROCEDURE insert_new_partition();


CREATE TABLE IF NOT EXISTS public.CASE_VAULT
(
    id serial,
    organization text,
    case_id varchar(16),
    create_time timestamp without time zone,
    user_id varchar(64),
    reporter text,
    title text,
    description text,
    is_pin bool DEFAULT false,
    s3_bucket text,
    share_token varchar(64),
    last_update timestamp default current_timestamp,
    case_id_alias text,
    CONSTRAINT pk_case_vault PRIMARY KEY (organization, case_id)
);

CREATE INDEX IF NOT EXISTS idx_case_vault_organization ON CASE_VAULT(organization);


CREATE TABLE IF NOT EXISTS public.CASE_VAULT_OBJECT
(
    organization text,
    case_id varchar(16),
    mac varchar(32) NOT NULL,
    oid bigint,
    note text,
    group_name text,
    camera_name text,
    create_time timestamp without time zone,
    obj_type varchar(16) NOT NULL,
    first_time timestamp without time zone,
    end_time timestamp without time zone,
    has_face bool DEFAULT false,
    has_license bool DEFAULT false,
    export_video_status varchar(32),
    CONSTRAINT pk_case_vault_object PRIMARY KEY (organization, case_id, mac, oid),
    CONSTRAINT fk_case_vault_object_case_vault FOREIGN KEY(organization, case_id)
    REFERENCES CASE_VAULT(organization, case_id) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_organization_case_object ON CASE_VAULT_OBJECT(organization);
CREATE INDEX IF NOT EXISTS idx_organization_case_id_case_object ON CASE_VAULT_OBJECT(organization, case_id);
CREATE INDEX IF NOT EXISTS idx_mac_oid_case_object ON CASE_VAULT_OBJECT(mac, oid);
CREATE INDEX IF NOT EXISTS idx_first_time_case_object ON CASE_VAULT_OBJECT(first_time);


CREATE TABLE IF NOT EXISTS public.ARCHIVE_OBJECT
(
    organization text,
    mac varchar(32) NOT NULL,
    oid bigint,
    thumbnail jsonb,
    background_thumbnail jsonb,
    archive_create_time timestamp without time zone,
    archive_start_time timestamp without time zone,
    archive_end_time timestamp without time zone,
    archive_status int DEFAULT 0,
    playback_video jsonb,
    content jsonb,
    CONSTRAINT pk_archive_object PRIMARY KEY (organization, mac, oid)
);

CREATE INDEX IF NOT EXISTS idx_organization_archive_object ON ARCHIVE_OBJECT(organization);
CREATE INDEX IF NOT EXISTS idx_mac_archive_object ON ARCHIVE_OBJECT(mac);
CREATE INDEX IF NOT EXISTS idx_oid_archive_object ON ARCHIVE_OBJECT(oid);
CREATE INDEX IF NOT EXISTS idx_archive_status_archive_object ON ARCHIVE_OBJECT(archive_status);


CREATE TABLE IF NOT EXISTS public.CASE_VAULT_EXPORT
(
    organization text,
    case_id varchar(16),
    export_type varchar(16),
    create_time timestamp without time zone,
    user_id varchar(64),
    creator text,
    s3_key text,
    watermark bool default FALSE,
    password text default NULL,
    status text,
    share_token varchar(64) default NULL,
    CONSTRAINT pk_case_vault_export PRIMARY KEY (organization, case_id, export_type),
    CONSTRAINT fk_case_vault_export_case_vault FOREIGN KEY(organization, case_id)
    REFERENCES CASE_VAULT(organization, case_id) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS public.DSLM_USER_HISTORY (
    id SERIAL PRIMARY KEY,
    user_id varchar(64) NOT NULL,
    query TEXT NOT NULL,
    metadata JSONB,
    embedding float[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_dslm_user_history_user_id ON DSLM_USER_HISTORY (user_id);

CREATE TABLE IF NOT EXISTS public.CAMERA_DSLM_DESCRIPTION (
    mac varchar(32) PRIMARY KEY NOT NULL REFERENCES CAMERA_INFO(mac) ON DELETE CASCADE,
    description TEXT,
    description_embedding float[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
