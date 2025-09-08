"""
Utility functions for domain handling in ReID tasks.
"""

def create_domain_mapping(source_domains, target_domains=None):
    """
    Create mapping from domain strings to integer labels.
    
    Args:
        source_domains (list): List of source domain names
        target_domains (list, optional): List of target domain names
        
    Returns:
        dict: Mapping from domain string to integer label
        
    Example:
        >>> create_domain_mapping(["tiger", "stoat"], ["market1501"])
        {'tiger': 0, 'stoat': 1, 'market1501': 2}
    """
    domain_mapping = {}
    
    # Assign source domains
    for i, domain in enumerate(source_domains):
        domain_mapping[domain] = i
    
    # Assign target domains (offset by source count)
    if target_domains:
        source_count = len(source_domains)
        for i, domain in enumerate(target_domains):
            domain_mapping[domain] = source_count + i
    
    return domain_mapping


def convert_domains_to_labels(domain_strings, domain_mapping):
    """
    Convert list of domain strings to integer labels.
    
    Args:
        domain_strings (list): List of domain names
        domain_mapping (dict): Mapping from domain string to integer
        
    Returns:
        list: List of integer domain labels
        
    Example:
        >>> domains = ["tiger", "stoat", "tiger"]
        >>> mapping = {"tiger": 0, "stoat": 1}
        >>> convert_domains_to_labels(domains, mapping)
        [0, 1, 0]
    """
    return [domain_mapping[domain] for domain in domain_strings]


def get_domain_mapping_from_config(cfg):
    """
    Create domain mapping from configuration.
    
    Args:
        cfg: Configuration object with DATASET.SOURCE_DOMAINS and DATASET.TARGET_DOMAINS
        
    Returns:
        dict: Domain string to integer mapping
    """
    source_domains = getattr(cfg.DATASET, 'SOURCE_DOMAINS', [])
    target_domains = getattr(cfg.DATASET, 'TARGET_DOMAINS', [])
    
    return create_domain_mapping(source_domains, target_domains)


def print_domain_mapping(domain_mapping, source_domains=None, target_domains=None):
    """
    Pretty print domain mapping for debugging.
    
    Args:
        domain_mapping (dict): Domain to label mapping
        source_domains (list, optional): Source domain names for labeling
        target_domains (list, optional): Target domain names for labeling
    """
    print("\nDomain Label Mapping:")
    print("-" * 40)
    
    if source_domains:
        print("Source Domains:")
        for domain in source_domains:
            if domain in domain_mapping:
                print(f"  {domain:>15} → domain_label = {domain_mapping[domain]}")
    
    if target_domains:
        print("Target Domains:")
        for domain in target_domains:
            if domain in domain_mapping:
                print(f"  {domain:>15} → domain_label = {domain_mapping[domain]}")
    
    if not source_domains and not target_domains:
        for domain, label in sorted(domain_mapping.items(), key=lambda x: x[1]):
            print(f"  {domain:>15} → domain_label = {label}")
    
    print("-" * 40)

