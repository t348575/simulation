use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, GenericParam};

#[proc_macro_derive(DNeuronInfo)]
pub fn derive_neuron_info(input: TokenStream) -> TokenStream {
    let input: DeriveInput = parse_macro_input!(input);
    let name_str = input.ident.to_string();
    let name = input.ident;

    let generic_names = input
        .generics
        .params
        .iter()
        .filter_map(|x| {
            if let GenericParam::Type(t) = x {
                Some(t)
            } else {
                None
            }
        })
        .map(|x| x.ident.clone())
        .collect::<Vec<_>>();
    let generics = input.generics;
    quote! {
        impl #generics NeuronInfo for #name<#(#generic_names, )*> {
            fn _type(&self) -> &'static str {
                #name_str
            }

            fn id(&self) -> usize {
                self.id
            }
        }
    }
    .into()
}

#[proc_macro_derive(SubTraits)]
pub fn derive_neuron_sub_traits(input: TokenStream) -> TokenStream {
    let input: DeriveInput = parse_macro_input!(input);
    let name = input.ident;

    let generic_names = input
        .generics
        .params
        .iter()
        .filter_map(|x| {
            if let GenericParam::Type(t) = x {
                Some(t)
            } else {
                None
            }
        })
        .map(|x| x.ident.clone())
        .collect::<Vec<_>>();
    let generics = input.generics;
    quote! {
        #[typetag::serde]
        impl #generics NeuronSubTraits for #name<#(#generic_names, )*> {}
    }
    .into()
}
